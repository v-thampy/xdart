"""The native RSM page (AUD-9, stage 1). The maintainer, 2026-09-30: "Can you start aud 9 in a separate
branch while codex reviews and I test". The RSM dialog's workflow in the main workspace on the shared
Controls panel, run bar and scientific panes, over the one RSM v2 owner."""
from __future__ import annotations

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from xdart.gui.pages.operation_owner import SingleWorkerOwner
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.rsm import page as page_module
from xdart.gui.tabs.rsm.page import RSMPage, suggested_output
from xdart.gui.tabs.rsm.display import rsm_keyed_projection
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_ENABLED, THRESHOLD_MAX, THRESHOLD_MIN
from xdart.gui.tools import rsm_owner
from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleTerminalResult
from xrd_tools.analysis.rsm_operation import RSMOperationExecutionV2, RSMOperationResultV2
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from tests.core.test_rsm_calibration_geometry import _canonical_partner
from tests.core.test_rsm_operation_r2_real_data import _SCIENCE, rsm_r2_project  # noqa: F401
from xdart.gui.pages.operation_owner import OperationIdentity, OperationTerminalStatus
from xdart.gui.pages.values import CloseReceipt
from xdart.gui.tools.rsm_owner import RSMOwnerAction, RSMOwnerOutcome, RSMOwnerOutcomeKind, RSMOwnerUpdate
from xdart.gui.tools.rsm_values import RSMScanMemberForm, RSMToolFormV2, rsm_tool_preset
from xrd_tools.analysis.rsm_geometry_asset import CANONICAL_RSM_GEOMETRY_LOCATOR, rsm_geometry_asset_input
from xrd_tools.analysis.rsm_operation import RSMNormalizationPolicy
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
from xrd_tools.session.rsm_viewer_model import make_rsm_viewer_values


# The doubles and synthetic values below were the classic RSM dialog's test helpers; they moved here when the
# dialog was retired (AUD-9 stage 2b G10).
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
        self.finalization = rsm_owner.RSMOwnerFinalization.NONE
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


def _frame_values(names, units, shape=(5, 7, 9), *, offset=0.0):
    axes = tuple((name, np.linspace(offset + index, offset + index + 1.0, size))
                 for index, (name, size) in enumerate(zip(names, shape, strict=True)))
    intensity = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=axes, axis_units=tuple(zip(names, units, strict=True)),
        intensity=intensity, sigma=None, coverage=None, normalization=None)
    return make_rsm_viewer_values(projection)


def _values(shape=(5, 7, 9), *, offset=0.0):
    return _frame_values(("h", "k", "l"), (None, None, None), shape, offset=offset)


def _q_values(shape=(5, 7, 9), *, offset=0.0):
    frame = RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    return _frame_values(frame.axis_names, frame.axis_units, shape, offset=offset)


def _scan43_dialog_form(root, output, *, frame=RSMCoordinateFrame.HKL, bins=None, normalization=None):
    """The scan-43 RSM v2 form the classic RSM dialog built (retired in G10), composed as it composed it from
    the shared preset: the Project as typed, the member's paths under it, the preset's conditioning, chunk and
    memory, CREATE_NEW and the canonical geometry asset; its grid choice (``bins``) and normalization choice."""
    preset = rsm_tool_preset()
    member = RSMScanMemberForm(root / preset.spec_relative_path, preset.scan,
                               root / preset.image_directory_relative_path, preset.image_stem,
                               preset.frame_selector, preset.detector_shape, preset.raw_dtype,
                               preset.raw_header_skip, preset.motor_selectors)
    return RSMToolFormV2(
        str(root), rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR), (member,), preset.conditioning,
        preset.normalization if normalization is None else normalization,
        preset.quick_bins if bins is None else bins, preset.chunk_size, preset.max_frame_bytes,
        preset.max_chunk_bytes, output, AnalysisArtifactOverwrite.CREATE_NEW, coordinate_frame=frame)



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
    until = time.monotonic() + 5
    receipt = page.close_page()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = page.close_page()
    assert receipt.status is PageCleanup.CLEAN
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


def test_the_page_builds_the_retired_dialogs_exact_scan43_form(qapp, tmp_path):
    """The page makes the exact RSM v2 form the classic RSM dialog made from the same inputs (its fingerprint
    names every member, selector, grid, frame and the output), for HKL and for Cartesian Q. The dialog is retired
    (G10); ``_scan43_dialog_form`` composes its form from the shared preset as it did, which G10 proved equal to
    the dialog's own before deleting it. The page's differences are deliberate: it replaces its own result
    (stage 2), its geometry is its Calibration row's (stage 2b C7b), its Max chunk MiB is Auto (S3a), and its
    threshold is Int's (Auto by default), the dialog's "High → NaN" retired into Int's Manual High (G8b), which
    ``_as_page`` applies."""
    from dataclasses import replace as replaced
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning

    def _as_page(form, **fields):
        return replaced(form, conditioning=replaced(form.conditioning, high_threshold=None),
                        corrections=replaced(form.corrections, threshold=StitchConditioning("auto")), **fields)
    page = _page(qapp, owner=_Owner())
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
        geometry = page.build_form().geometry_asset
        assert page.build_form().fingerprint == _as_page(
            _scan43_dialog_form(tmp_path, output), overwrite=AnalysisArtifactOverwrite.REPLACE,
            geometry_asset=geometry, max_chunk_bytes=None).fingerprint
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page.set_field("normalization", "None")
        assert page.build_form().fingerprint == _as_page(
            _scan43_dialog_form(tmp_path, output, frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
                                bins=rsm_tool_preset().full_bins, normalization=RSMNormalizationPolicy.identity()),
            overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=geometry, max_chunk_bytes=None).fingerprint
    finally:
        _close(page, qapp)


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


def test_rsm_source_counts_match_stitch_and_follow_the_selected_scan(qapp, tmp_path):
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert "4 frames" in page.controls.source_card.status.text()
        assert "4 frames" in page.run_bar.readinessLabel.full_text()
        assert "scan" in page.run_bar.readinessLabel.toolTip()
        page.set_field("scan", "1")
        assert "2 SPEC points" in page.controls.source_card.status.text()
        page.resolve_source()
        assert "2 frames" in page.controls.source_card.status.text()
        assert "2 frames" in page.run_bar.readinessLabel.full_text()
    finally:
        _close(page, qapp)


def test_rsm_grid_uses_independent_axis_counts_and_keeps_legacy_presets(qapp, tmp_path):
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        for axis, count in zip("hkl", (12, 16, 20), strict=True):
            page.set_field(f"grid_{axis}", str(count))
        assert page.build_form().bins == (12, 16, 20)
        assert "grid 12×16×20" in page.run_bar.readinessLabel.full_text()
        assert [page._rows[f"grid_{axis}"].label.text() for axis in "hkl"] == ["H", "K", "L"]
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        assert [page._rows[f"grid_{axis}"].label.text() for axis in "hkl"] == ["Qx", "Qy", "Qz"]
        assert page.build_form().bins == (12, 16, 20)
        assert all(page._rows[f"grid_{axis}"].toolTip() for axis in "hkl")
        page.set_field("grid_h", "0")
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="Grid"):
            page.build_form()
        page.set_field("grid", "Full 200×200×200")
        assert page.build_form().bins == (200, 200, 200)
        page.set_field("grid_l", "201")
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="8,000,000"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_rsm_notebook_conditioning_is_advanced_and_explained(qapp):
    page = _page(qapp)
    try:
        for key in ("offset", "static_hot"):
            assert page.advanced.isAncestorOf(page._rows[key])
            assert not page.controls.processing_card.isAncestorOf(page._rows[key])
            assert page._rows[key].editor.toolTip()
        assert "unchanged" in page._rows["static_hot"].toolTip()
        assert "0 disables" in page._rows["offset"].toolTip()
        offset_tip = page._rows["offset"].toolTip()
        page.set_field("offset", "0")
        assert all(widget.toolTip() == offset_tip for widget in
                   (page._rows["offset"], page._rows["offset"].editor, page._rows["offset"].label))
        assert "The UB an HKL map uses" in page._rows["ub_from"].toolTip()
        assert "without gridding again" in page.retry_button.toolTip()
    finally:
        _close(page, qapp)


def test_rsm_opens_with_six_empty_native_panels_and_labels(qapp):
    """Selecting RSM shows its own layout before any source or result is loaded."""
    page = _page(qapp)
    try:
        assert page.shown is None
        assert page.view.share_axis.isChecked() and page.view.keyed_projection.share_axis
        assert page.view.vertical_splitter.isHidden()
        assert [key.instance for key in _keys(page.view)] == ["HK", "KL", "LH", "H", "K", "L"]
        assert all(not pane.canvas.raw_image.size for pane in _slices(page.view))
        assert all(selector.isHidden() for selector in _selectors(page.view))
        assert all(curve.getData()[0] is None or not curve.getData()[0].size for curve in _curves(page.view))
        assert page.view.keyed_panel(_keys(page.view)[0]).pane.plot.getAxis("bottom").labelText == "H"
        page.view.color_map.setCurrentText("magma")
        page.view.log_scale.click()
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        assert [key.instance for key in _keys(page.view)] == ["QxQy", "QyQz", "QzQx", "Qx", "Qy", "Qz"]
        assert page.view.keyed_panel(_keys(page.view)[0]).pane.plot.getAxis("bottom").labelText == (
            RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.axis_labels[0])
        assert page.view.color_map.currentText() == "magma" and page.view.log_scale.isChecked()
        assert page.shown is None
    finally:
        _close(page, qapp)


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
        assert page._map_choice.currentText() == "Mean over range"
        page.view._show_plot_options()
        page._map_choice.setCurrentText("Single plane")
        page.view.plot_options_dialog.accept_button.click()
        snapshot = page.shown[1]
        for pane, product in zip(_slices(page.view), snapshot.products[:3], strict=True):
            np.testing.assert_array_equal(pane.canvas.raw_image, np.asarray(product.values).T)
        for curve, product in zip(_curves(page.view), snapshot.products[3:], strict=True):
            x, y = curve.getData()
            np.testing.assert_array_equal(x, product.x_axis)
            np.testing.assert_array_equal(y, product.values)
        assert [selector.currentIndex() for selector in _selectors(page.view)] == [4, 2, 3]  # L, H, K
        _selectors(page.view)[1].setCurrentIndex(0)  # the KL slice's H index: it moves to H 0
        moved = page.shown[1]
        assert (moved.state.h_index, moved.state.k_index, moved.state.l_index) == (0, 3, 4)
        np.testing.assert_array_equal(_slices(page.view)[2].canvas.raw_image, np.asarray(moved.products[2].values).T)
        assert len(set(page.browser.listed_results()) & set(page._listed)) == 1
        assert Path(page.browser.current_result()).name == output.name
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
        assert [Path(path).name for path in page.browser.listed_results() if path in page._listed] == [f"{name}_rsm.nexus" for name in shown]
        page.browser.set_current_result(str(tmp_path / 'two_rsm.nexus'), keep_selection=False)  # kept: shown at once
        assert page.shown[0] is shown["two"] and len(reads) == 6
        page.browser.set_current_result(str(tmp_path / 'one_rsm.nexus'), keep_selection=False)  # the oldest: read again
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
    """Each axis is labelled as its frame names it, as the classic RSM dialog did (a Cartesian Q axis as Qx,
    Qy or Qz in Å⁻¹, not as Int's 1-D "Q"), and each slice title names the index it is taken at."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    view = ScientificView()
    try:
        values = make()
        view.reconcile_keyed(rsm_keyed_projection(values, RSMViewerModel().snapshot(values)))
        first, second, third = names
        assert [(pane.plot.getAxis("bottom").labelText, pane.plot.getAxis("left").labelText)
                for pane in _slices(view)] == [(first, second), (second, third), (third, first)]
        assert [plot.getAxis("bottom").labelText for plot in _plots(view)] == list(names)
        assert {pane.plot.getAxis(side).labelUnits for pane in _slices(view) for side in ("bottom", "left")} | {
            plot.getAxis("bottom").labelUnits for plot in _plots(view)} == {""}
        symbols = values.coordinate_frame.axis_symbols
        normal = float(np.asarray(values.axes[2][1])[4])
        assert _slices(view)[0].plot.titleLabel.text == (
            f"{symbols[0]}{symbols[1]} slice · {symbols[2]} = {normal:.3f} (index 4)")
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


@pytest.mark.slow
def test_scan43_from_a_rough_calibration_authored_in_the_window_runs_and_reopens(  # noqa: F811
        qapp, rsm_r2_project, monkeypatch):
    """Row C (AUD-9 stage 2b item 9): a rough (2 · Rough geometry) calibration of STO scan 43 authored in the shared
    calibration window opened from the page, for RSM's Pilatus 100K (camera z−, x−: stored 195 × 487, turned 270°
    in the window): row B's distance and beam pixel typed, saved and adopted on the Calibration row. Preview names
    it a direct-beam calibration (item 7); the page grids scan 43 at 40³, commits and reopens it.

    Tolerance: a rough geometry is the distance and beam pixel with the rotations held, which here are row B's
    canonical geometry (its offsets and tilts are zero), so the result must be row B's: the grid axes and the
    finite voxels identical (within rtol 1e-8, the transfer gate of the window's partner, 1e-8 Å⁻¹), with every
    correction off the reference fingerprint pinned for row B exactly, and with the page's corrections (solid
    angle, polarization 0.99) the intensities within rtol 1e-6 of row B's (the partner's polarization gate; the
    solid angle and polarization come from the partner's own arm, 1.0e-7 at most on 3 of 5531 voxels when
    measured). Its grid spans the (1 1 3) neighbourhood scan 43 was taken around."""
    import json
    from dataclasses import replace
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.core.geometry.camera import DetectorCamera
    from xrd_tools.integrate.stitch_calibration import xu_partner_path
    project = Path(rsm_r2_project)
    canonical = _canonical_partner(project)  # row B's calibration, the reference
    target = (project / "calibration" / "rowc_rough.json").resolve()
    page = _page(qapp)
    try:
        page.set_field("project_root", str(project))
        page.set_sources(["STO_align"])
        page.set_field("scan", "43")
        page.set_field("image_dir", "images")
        page._calibrate()
        dialog = page._calibration_dialog
        # The window reads the page's scan metadata as it opens (since ebf112a4), its tabs waiting meanwhile: wait
        # as a person does before typing (RSM-ROUGH: a click during that read did nothing).
        _wait(page, qapp, lambda: not dialog.active(), seconds=60)
        dialog.camera.set_value(DetectorCamera("z-", "x-"))
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData("manual"))
        for key, value in (("beam_detector", "Pilatus100k"), ("rough_distance", "1014.7173"),
                           ("beam_pixel", "97, 243"),  # row B's (row 97, column 243) stored, turned: column, row
                           ("rough_energy", "13.000007")):
            dialog._set_field_text(key, value)
        hint = dialog._image_size.text()  # said last: the window's frame (item 8)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *args, **kwargs: (str(target), "")))
        dialog.save_rough_button.click()
        deadline = time.monotonic() + 120
        while (dialog.active() or page._value("geometry_path") != str(target)) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert page._value("geometry_path") == str(target), dialog.status.text()
        dialog.close()
        rough = json.loads(xu_partner_path(target).read_text(encoding="utf-8"))["detector"]
        reference = json.loads((project / canonical).with_name("rsm_canonical.xu.json").read_text())["detector"]
        assert (rough["detector_factory_name"], rough["shape"]) == ("Pilatus100k", reference["shape"])
        assert (rough["cch1_px"], rough["cch2_px"]) == pytest.approx((reference["cch1_px"], reference["cch2_px"]),
                                                                     abs=1e-9)
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text(), seconds=120)
        assert ("  calibration: direct-beam (rough): the distance and beam pixel, rotations held; not ring-fitted"
                in page.preview_text.toPlainText().splitlines())
        page.begin_run()
        _wait(page, qapp, lambda: not page.active(), seconds=600)
        output = page._value("output_path")
        assert page.run_bar.readinessLabel.full_text() == f"Saved {Path(output).name}"
        committed = page._results[output]
        assert page.reopen(output)
        _wait(page, qapp, lambda: not page.active())
        reopened = page._results[output]
        assert reopened is not committed and reopened.result_fingerprint == committed.result_fingerprint
        np.testing.assert_array_equal(page.shown[0].intensity, committed.intensity)
        # row B's, headless from the page's own form: every correction off (its pinned reference), then the page's
        rough_form = page.build_form()
        headless = project / "rowc-headless.nexus"
        off = run_rsm_operation_v2(prepare_rsm_tool_v2(replace(
            rough_form, output_path=headless, output_family=None, corrections=RSMFrameCorrections())).request)
        assert off.payload.result_fingerprint == _SCIENCE[((43,), 40)]["result"]
        page.set_field("geometry_path", canonical)
        page.resolve_source()
        row_b = run_rsm_operation_v2(prepare_rsm_tool_v2(replace(
            page.build_form(), output_path=headless.with_name("rowc-row-b.nexus"), output_family=None)).request).payload
    finally:
        _close(page, qapp)
    finite = np.isfinite(committed.intensity)
    assert int(finite.sum()) == int(np.isfinite(row_b.intensity).sum()) == _SCIENCE[((43,), 40)]["finite"]
    np.testing.assert_array_equal(finite, np.isfinite(row_b.intensity))
    for (name, values), (reference_name, reference_values) in zip(committed.axes, row_b.axes, strict=True):
        assert name == reference_name
        np.testing.assert_allclose(values, reference_values, rtol=1e-8, atol=0)
    np.testing.assert_allclose(committed.intensity[finite], row_b.intensity[finite], rtol=1e-6, atol=0)
    spans = {name: (float(values[0]), float(values[-1])) for name, values in committed.axes}
    assert spans["h"][0] <= 1 <= spans["h"][1] and spans["k"][0] <= 1 <= spans["k"][1]
    assert spans["l"][0] <= 3 <= spans["l"][1]
    assert hint == "195 × 487 px (W × H)"


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
                              "Chunk and memory", "Image conditioning", "Images per point"}
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
    bins' outer edges (the classic dialog's half-bin bounds, now one headless helper): each pixel centre is its bin
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
        path = tmp_path / f"r{index}.nexus"
        write_record(root, False, target=path)
        paths.append(str(path))
    started, release, hold, reads = Event(), Event(), [], []

    def read(path, cancel, progress):
        reads.append(path)
        if hold and path == paths[0]:
            started.set()
            assert release.wait(10)
        return page_module._read_rsm_record(path, cancel, progress)

    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    def select(path):
        page.results.setCurrentItem(next(page.results.item(index) for index in range(page.results.count())
            if page.results.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == path))
    try:
        for path in paths:
            assert page.reopen(path)
            _wait(page, qapp, lambda: not page.active())
        assert page._results[paths[0]] is page._results[paths[1]] is None
        hold.append(True)
        select(paths[0])
        assert started.wait(3)
        select(paths[1])
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


def test_a_catalog_result_opens_through_the_native_rsm_reader(qapp, tmp_path):
    from tests.core.test_analysis_artifact import (
        test_analysis_artifact_v2_rsm_round_trip_is_a_closed_exact_branch as write_record,
    )
    write_record(tmp_path, False)
    path = str(tmp_path / "rsm-artifact-v2.nexus")
    page = _page(qapp)
    try:
        page._result_browser.set_directory(str(tmp_path))
        _wait(page, qapp, lambda: not page._result_browser.busy)
        assert path in page.browser.listed_results()
        assert page.browser.current_result() is None and page.shown is None
        page.browser.set_current_result(path, keep_selection=False)
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        assert page.browser.current_result() == path and page._shown_path == path
        assert _curves(page.view) and all(curve.getData()[0] is not None for curve in _curves(page.view))
    finally:
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
        path = tmp_path / f"r{index}.nexus"
        write_record(root, False, target=path)
        paths.append(str(path))
    started, released, hold = Event(), Event(), []

    def read(path, cancel, progress):
        if hold and path == paths[0]:
            started.set()
            assert released.wait(10), "the held read was not released"
        return page_module._read_rsm_record(path, cancel, progress)

    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    role = QtCore.Qt.ItemDataRole.UserRole
    def select(path):
        page.results.setCurrentItem(next(page.results.item(index) for index in range(page.results.count())
            if page.results.item(index).data(role) == path))
    try:
        for path in paths:
            assert page.reopen(path)
            _wait(page, qapp, lambda: not page.active())
        assert page._results[paths[0]] is None and page._results[paths[1]] is not None
        hold.append(True)
        select(paths[0])
        assert started.wait(3)
        select(paths[1])
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


@pytest.mark.parametrize("scans", ("1", "1,2"))
def test_append_on_a_blank_page_displays_the_existing_rsm(qapp, tmp_path, scans):
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2

    _synthetic_source(tmp_path, full=True)
    page = _page(qapp)
    try:
        _with_source(page, tmp_path, scans=scans)
        _wait(page, qapp, lambda: not page.active())
        forms = page.build_forms()
        assert len(forms) == (1 if scans == "1" else 2)
        records = []
        for form in forms:
            result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
            assert result.terminal.disposition is ModuleDisposition.COMMITTED
            target = Path(form.output_path)
            records.append((target, target.stat().st_ino, target.stat().st_mtime_ns,
                            target.read_bytes(), result.payload.result_fingerprint))
        assert page.shown is None
        page.run_bar.startButton.click()
        _wait(page, qapp, lambda: not page.active(), seconds=60)
        assert page.shown is not None
        assert page.shown[0].result_fingerprint == records[-1][-1]
        assert _slices(page.view)[0].canvas.raw_image.size > 0
        assert "already gridded with these settings" in page.run_bar.readinessLabel.full_text()
        for target, inode, mtime, data, _fingerprint in records:
            assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == (inode, mtime, data)
    finally:
        _close(page, qapp)


def test_append_skips_an_up_to_date_output_and_replace_grids_it_again(qapp, tmp_path):
    """Stitch's semantics: the run bar's toggle (Append by default) is live. In Append a Run skips an output
    whose stored result is this request's (the preflight's rsm_output_current) and says so; a stale one is
    gridded again; in Replace every Run grids. The form is REPLACE throughout: Append is the page's skip."""
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path, full=True)
        _with_source(page, tmp_path, scans="1")
        output = Path(page._value("output_path"))
        toggle = page.run_bar.writeModeButton
        assert toggle.isEnabled() and page.run_bar.write_mode() == "Append"
        assert toggle.toolTip().startswith("Append: skip the output")
        page.begin_run()
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        revision = (output.stat().st_ino, output.stat().st_mtime_ns, output.read_bytes())
        fingerprint = page.shown[0].result_fingerprint
        page.begin_run()
        _wait(page, qapp, lambda: "Up to date" in label.full_text() and not page.active())
        assert (output.stat().st_ino, output.stat().st_mtime_ns, output.read_bytes()) == revision
        assert page.shown[0].result_fingerprint == fingerprint
        assert label.full_text() == (
            f"Up to date · {output.name} already gridded with these settings (Replace ⇄ grids again)")
        assert "already gridded with these settings (Append skips it)" in page.preview_text.toPlainText()
        page.set_field("grid_h", 32)  # changed settings: Append grids it again
        page.begin_run()
        _wait(page, qapp, lambda: page.shown[0].shape == (32, 40, 40) and not page.active())
        assert output.stat().st_ino != revision[0]
        changed = output.stat().st_ino
        toggle.click()  # Replace
        assert page.run_bar.write_mode() == "Overwrite" and toggle.toolTip().startswith("Replace: grid again")
        page.begin_run()
        _wait(page, qapp, lambda: output.stat().st_ino != changed and not page.active())
        assert page.build_form().overwrite is AnalysisArtifactOverwrite.REPLACE
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
        assert len(set(page.browser.listed_results()) & set(page._listed)) == 1
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
        assert page.shown is prior and len(set(page.browser.listed_results()) & set(page._listed)) == 1
        _terminal_instead_of_gridding(monkeypatch, ModuleDisposition.CANCELLED, "CANCELLED", until_stopped=True)
        page.begin_run()
        _wait(page, qapp, lambda: page._owner.busy and page.run_bar.stopButton.isEnabled())
        page.stop()
        _wait(page, qapp, lambda: "stopped" in label.full_text() and not page.active())
        assert page.shown is prior and len(set(page.browser.listed_results()) & set(page._listed)) == 1
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
        assert page.run_bar.readinessLabel.full_text().startswith("1 SPEC file · 1 RSM")
        assert all(card.valid_marker.isVisibleTo(page) for card in cards.values())
        assert cards["source"].status.text() == "1 SPEC file · 1 RSM · 4 frames"  # Stitch's Source status
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


def test_ints_intensity_controls_and_share_axis_drive_the_six(qapp, monkeypatch):
    """G13 unit 1 (decision 1's toolbar: "colour map, Log, intensity, Set BG ... and Share Axis"): under the six,
    Int's intensity controls set one set of levels for the three slices, and Share Axis, a command the page keeps
    and draws back, links each coordinate across the panels: H on the HK and LH slices and the H projection."""
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", {"payload": _values()}.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/x_rsm.nexus", "payload")
        view = page.view
        levels = {tuple(pane.canvas.histogram.levels()) for pane in _slices(view)}
        assert view.viewer_intensity.percentile_values() == (0.5, 99.5)
        mapped = [np.nanpercentile(pane.canvas.displayed_image, (0.5, 99.5)) for pane in _slices(view)]
        span = (min(pair[0] for pair in mapped), max(pair[1] for pair in mapped))
        assert len(levels) == 1
        np.testing.assert_allclose(levels.pop(), span)
        assert not view.viewer_intensity.isHidden() and not view.share_axis.isHidden()
        assert view.keyed_projection.share_axis and view.share_axis.isChecked()
        hk, _kl, lh = _slices(view)
        hk.canvas.imageViewBox.setXRange(0.2, 0.6, padding=0)
        h_projection = next(panel for panel in view.keyed_projection.panels if not hasattr(panel, "image"))
        h_plot = view.keyed_panel(h_projection.key).plot.getViewBox()
        for box, axis in ((lh.canvas.imageViewBox, 1), (h_plot, 0)):
            np.testing.assert_allclose(box.viewRange()[axis], (0.2, 0.6))
        view.share_axis.click()
        assert not view.keyed_projection.share_axis
        hk.canvas.imageViewBox.setXRange(0.3, 0.5, padding=0)
        np.testing.assert_allclose(lh.canvas.imageViewBox.viewRange()[1], (0.2, 0.6))
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
        from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _wait as wait_calibration
        wait_calibration(dialog)
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


def test_rsms_calibration_window_starts_on_rsms_detector_and_camera_unless_saved(qapp, tmp_path):
    """Item 10 (the maintainer: "is this the camera orientation that works for the test data? If yes, then start
    with this orientation"): with no saved window settings RSM's calibration window starts on the canonical
    geometry's detector, the Pilatus 100K (195 × 487 at 0.172 mm), and its camera z−, x− (row C reproduces scan
    43's reference with them), its hint in the turned frame; the page's source still seeds the window but no
    longer turns its camera back. Saved window settings win, their detector and camera too."""
    import json
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.rsm_geometry_asset import canonical_rsm_geometry_resource_bytes
    from xrd_tools.core.geometry.camera import DetectorCamera
    from xrd_tools.session.readiness import ControlAction
    canonical = json.loads(canonical_rsm_geometry_resource_bytes())
    header, model = canonical["detector"]["header"], detector_factory("Pilatus100k")
    assert tuple(model.shape) == (header["Nch1"], header["Nch2"])
    assert (model.pixel1 * 1e3, model.pixel2 * 1e3) == pytest.approx((header["pwidth1"], header["pwidth2"]),
                                                                     rel=1e-12, abs=0)
    page, restored = _page(qapp), _page(qapp)
    try:
        _synthetic_source(tmp_path)
        for each in (page, restored):
            _with_source(each, tmp_path)
        page._control_action(ControlAction.CALIBRATE)
        dialog = page._calibration_dialog
        assert dialog._text("source") == str(tmp_path / "source" / "RSM_synth")
        assert (dialog._text("beam_detector"), tuple(dialog.camera.value().to_xu())) == (
            "Pilatus100k", tuple(canonical["diffractometer"]["camera"]))
        assert dialog._text("shape") == "195, 487"
        assert dialog._image_size.text() == "195 × 487 px (W × H)"
        dialog._set_field_text("beam_detector", "Pilatus300kw")
        dialog.camera.set_value(DetectorCamera("x-", "z-"))  # a mirror, the saved window's own
        saved = page.settings()
        saved["calibration"]["fields"]["source"] = ""  # saved without a source: the page's source seeds it again
        dialog.hide()
        restored.apply_settings(saved)
        restored._control_action(ControlAction.CALIBRATE)
        again = restored._calibration_dialog
        assert (again._text("beam_detector"), again.camera.value()) == ("Pilatus300kw", DetectorCamera("x-", "z-"))
        assert again._text("source") == str(tmp_path / "source" / "RSM_synth")
        again.hide()
    finally:
        for each in (page, restored):
            _close(each, qapp)


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
        assert page.controls.source_card.status.text() == "1 SPEC file · 2 RSMs · 4 frames"
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
        assert page._readiness[1] and page.controls.source_card.status.text() == "1 SPEC file · all scans · 4 SPEC points"
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
        assert grids == names and len(set(page.browser.listed_results()) & set(page._listed)) == 2 and not page.active()
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


# AUD-9 stage 2b G11: a queued Run checks every output before its first RSM is gridded


def test_a_queued_run_refuses_a_foreign_file_at_any_output_before_gridding_the_first(qapp, tmp_path, monkeypatch):
    """The maintainer, 2026-10-01, on queued runs: "Yes for your recommendation": every queued output is checked
    (from its header, while the scans are found, off the GUI thread) before the first RSM is gridded, as Stitch
    checks its outputs. A foreign file at RSM 2's output refuses the whole Run, named, with nothing gridded or
    saved and the file untouched; before, RSM 1 was gridded and saved, then RSM 2 refused."""
    grids = _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        second = Path(page.build_forms()[1].output_path)
        second.parent.mkdir(parents=True, exist_ok=True)
        second.write_bytes(b"an operator's own file")
        page.begin_run()
        _wait(page, qapp, lambda: not page.active() and "refused" in label.full_text())
        assert label.full_text().startswith("Run refused (RSM 2 of 2): OUTPUT_NOT_PREVIOUS_ANALYSIS: ")
        assert f"{second.name} is not an RSM result that Replace can overwrite (" in label.full_text()
        assert label.full_text().endswith(" · nothing was gridded")
        assert grids == [] and page.results.count() == 0 and page._jobs == []
        assert second.read_bytes() == b"an operator's own file"
    finally:
        _close(page, qapp)


def test_a_file_appearing_at_a_queued_output_after_the_check_is_refused_at_its_turn(qapp, tmp_path, monkeypatch):
    """The early check does not replace the one just before each RSM's gridding: a file that appears at RSM 2's
    output while RSM 1 is gridded is still refused by RSM 2's own preflight, and left untouched; RSM 1 is saved."""
    _commit_instead_of_gridding(monkeypatch, _values())
    committed, grids, planted = RSMOperationExecutionV2.run, [], []
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        first, second = (Path(form.output_path) for form in page.build_forms())

        def gridding(self, **kwargs):
            grids.append(Path(self.request.module.output.target).name)
            if not planted:  # RSM 1 is being gridded: the file appears at RSM 2's output
                second.parent.mkdir(parents=True, exist_ok=True)
                second.write_bytes(b"an operator's own file")
                planted.append(second)
            return committed(self, **kwargs)
        monkeypatch.setattr(RSMOperationExecutionV2, "run", gridding)
        page.begin_run()
        _wait(page, qapp, lambda: not page.active() and "refused" in label.full_text())
        assert label.full_text().startswith("Preview refused (RSM 2 of 2): ")
        assert f"{second.name} is not an RSM result" in label.full_text()
        assert grids == [first.name] and len(set(page.browser.listed_results()) & set(page._listed)) == 1
        assert second.read_bytes() == b"an operator's own file"
    finally:
        _close(page, qapp)


def test_a_save_path_changed_while_the_scans_were_found_asks_for_another_run(qapp, tmp_path, monkeypatch):
    """The outputs checked are the outputs written (``queue_output_slots`` serves both). When the Save Path moves
    while the scans are being found, the outputs checked were the old folder's: the queued Run grids nothing and
    asks for another Run, which checks and grids the new folder's."""
    from xdart.gui.tools import rsm_values
    grids = _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        found, release = rsm_values.rsm_member_groups, threading.Event()

        def held(*args, **kwargs):  # the scans are being found while the Save Path moves
            release.wait(10)
            return found(*args, **kwargs)
        monkeypatch.setattr(rsm_values, "rsm_member_groups", held)
        page.begin_run()
        elsewhere = tmp_path / "elsewhere"
        page.set_field("output_path", str(elsewhere / "named.nexus"))
        release.set()
        _wait(page, qapp, lambda: not page.active() and "Run again" in label.full_text())
        assert label.full_text() == "The output changed while the scans were found: Run again"
        assert grids == [] and page.results.count() == 0
        names = [Path(form.output_path) for form in page.build_forms()]
        assert [path.parent for path in names] == [elsewhere, elsewhere]
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 2 of 2 RSMs" in label.full_text())
        assert grids == [path.name for path in names]
    finally:
        _close(page, qapp)


def test_a_queued_preview_reads_no_output_header_ahead_of_its_rsms_preflights(qapp, tmp_path, monkeypatch):
    """G11 self-review: only a queued Run checks its outputs while the scans are found. A Preview grids nothing,
    and each RSM's own preflight checks its output, so the early check would read every header twice."""
    from xrd_tools.io import analysis_artifact
    _commit_instead_of_gridding(monkeypatch, _values())
    early, check = [], analysis_artifact.analysis_output_refusal

    def counted(target, kind):  # the early check's import (each preflight holds rsm_values' own)
        early.append(Path(target).name)
        return check(target, kind)
    monkeypatch.setattr(analysis_artifact, "analysis_output_refusal", counted)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        names = [Path(form.output_path).name for form in page.build_forms()]
        early.clear()  # resolve_source checks them (scripts and rows)
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready · 2 RSMs" in label.full_text())
        assert early == []
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 2 of 2 RSMs" in label.full_text())
        assert early == names
    finally:
        _close(page, qapp)


def _own_ubs(root, ubs):
    """The synthetic source's SPEC file again, each scan with its own #G3 (None: the scan records no UB)."""
    _spec_file(root / "source" / "RSM_synth", [
        _spec_scan(scan, 9 + scan).replace(
            "#G3 1 0 0 0 1 0 0 0 1\n", "" if ub is None else "#G3 " + " ".join(f"{value:g}" for value in ub) + "\n")
        for scan, ub in zip((1, 2), ubs, strict=True)])


def _gridded_ubs(monkeypatch):
    """The committing stand-in, recording each gridded RSM's members' UBs (nine numbers each) by output name."""
    _commit_instead_of_gridding(monkeypatch, _values())
    committed, used = RSMOperationExecutionV2.run, {}

    def gridding(self, **kwargs):
        used[Path(self.request.module.output.target).name] = [
            tuple(float(value) for value in np.asarray(member.ub, dtype=float).ravel())
            for member in self.request.preflight.members]
        return committed(self, **kwargs)
    monkeypatch.setattr(RSMOperationExecutionV2, "run", gridding)
    return used


def test_a_queued_run_grids_each_rsm_with_its_members_own_ub_and_waits_for_any_missing_one(
        qapp, tmp_path, monkeypatch):
    """G8b's UB rule over G8c's queue (the choices compilation's item 52, G8b's choice 10: "the page grids one RSM
    per run" no longer holds). In HKL with the source's own UB, each queued RSM is gridded with its own members'
    UB; a member of any RSM that records none (RSM 2's, then RSM 1's) holds the whole queue, warned, before any
    gridding."""
    used = _gridded_ubs(monkeypatch)
    own = ((1, 0, 0, 0, 1, 0, 0, 0, 1), (2, 0, 0, 0, 2, 0, 0, 0, 2))
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _own_ubs(tmp_path, own)
        _with_source(page, tmp_path, scans="1,2")
        first, second = (Path(form.output_path).name for form in page.build_forms())
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 2 of 2 RSMs" in label.full_text())
        assert used == {first: [tuple(map(float, own[0]))], second: [tuple(map(float, own[1]))]}
        for ubs, warning in (((own[0], None), "Scan 2 records no UB matrix. " + _NO_UB_CHOICES),
                             ((None, own[1]), "Scan 1 records no UB matrix. " + _NO_UB_CHOICES)):
            used.clear()
            _own_ubs(tmp_path, own)
            page.resolve_source()  # every member records one again: ready
            assert page._readiness[1], page._readiness
            _own_ubs(tmp_path, ubs)
            page.begin_run()  # the scans are found again, the member recording none with them, which is named
            _wait(page, qapp, lambda: not page.active())
            assert label.full_text() == warning and used == {} and page._jobs == []
            assert page._headers["source"].detail == warning and not page._readiness[1]
            page.begin_run()  # and Run waits while it records none
            assert label.full_text() == "Needs a UB matrix" and not page.active() and used == {}
    finally:
        _close(page, qapp)


def test_a_ub_given_for_a_queued_run_is_every_rsms_and_the_preview_names_each_members_own(
        qapp, tmp_path, monkeypatch):
    """A UB given (typed, or a SPEC scan's) is the run's, as the design's override rule has it ("An override is
    shown in the Preview beside the source's own UB"): every queued RSM is gridded with it, the one whose member
    records its own UB too, and each RSM's Preview names its members' own beside it."""
    used = _gridded_ubs(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _own_ubs(tmp_path, ((2, 0, 0, 0, 2, 0, 0, 0, 2), None))
        _with_source(page, tmp_path, scans="1,2")
        assert page._headers["source"].detail == "Scan 2 records no UB matrix. " + _NO_UB_CHOICES
        page.set_field("ub_from", "Typed")
        for number, row in enumerate(("3 0 0", "0 3 0", "0 0 3"), 1):
            page.set_field(f"ub_row{number}", row)
        assert page._readiness[1], page._readiness
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready · 2 RSMs" in label.full_text())
        shown = page.preview_text.toPlainText()
        assert shown.count("UB given: typed, in SPEC's #G3 convention: 3 0 0; 0 3 0; 0 0 3, in place of the "
                           "sources' own (each member's below)") == 2
        assert shown.index("UB the source records: 2 0 0; 0 2 0; 0 0 2") < shown.index("RSM 2 of 2") \
            < shown.index("UB the source records: none recorded")
        names = [Path(form.output_path).name for form in page.build_forms()]
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 2 of 2 RSMs" in label.full_text())
        assert used == dict.fromkeys(names, [(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0)])
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
        assert "5 frames" in page.controls.source_card.status.text()
        assert "5 frames" in page.run_bar.readinessLabel.full_text()
        for key, count in zip(("grid_h", "grid_k", "grid_l"), (12, 16, 20), strict=True):
            page.set_field(key, count)
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        member, = page.build_form().members
        assert type(member) is RSMNexusMemberForm and member.entry == "entry"
        assert "member_ub.nxs · NeXus entry entry · 5 frames" in page.preview_text.toPlainText()
        page.begin_run()  # gridded for real: five frames
        _wait(page, qapp, lambda: page.shown is not None and not page.active(), seconds=300)
        assert page.shown[0].shape == (12, 16, 20)
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


def test_conditioning_editors_align_with_threshold(qapp, tmp_path):
    page = _page(qapp)
    try:
        page.set_field("project_root", str(tmp_path))
        page.resize(1400, 1000)
        page.show()
        for mode in ("Foil transmission × exposure", "Monitor × foil transmission"):
            page.set_field("normalization", mode)
            qapp.processEvents()
            left = lambda editor: editor.mapTo(page.controls, QtCore.QPoint(0, 0)).x()
            expected = left(page.threshold_range._low)
            positions = {key: left(page._rows[key].editor) for key in (
                "normalization", "monitor", "foil_column",
                "exposure_column", "foil_lengths") if not page._rows[key].isHidden()}
            assert set(positions.values()) == {expected}, (expected, positions)
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
        assert page._rows['normalization'].label.text() == 'Normalize'
        assert page._rows['monitor'].isHidden()
        assert all(not page._rows[key].isHidden() for key in ('foil_column', 'exposure_column', 'foil_lengths'))
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
        assert not page._rows['monitor'].isHidden()
        assert all(page._rows[key].isHidden() for key in ('foil_column', 'exposure_column', 'foil_lengths'))
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
        assert page._rows['exposure_column'].isHidden()
        assert all(not page._rows[key].isHidden() for key in ('monitor', 'foil_column', 'foil_lengths'))
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
        for key, value in (("dark_path", "dark.npy"), ("flat_path", "flat.npy"), ("flat_cutoff", "20"),
                           ("polarization", ""), ("solid_angle", "Off")):
            page.set_field(key, value)
        assert page._readiness[1], page._readiness
        corrections = page.build_form().corrections
        assert (corrections.dark_path, corrections.flat_path) == (str(tmp_path / "dark.npy"),
                                                                  str(tmp_path / "flat.npy"))
        assert (corrections.flat_min_fraction, corrections.polarization_factor, corrections.solid_angle) == (
            0.2, None, False)
        page.set_field("polarization", "2")
        assert "Advanced Polarization" in page._headers["processing"].detail and not page._readiness[1]
        with pytest.raises(ValueError, match="Polarization"):
            page.build_form()
        page.set_field("polarization", "0.5")
        page.set_field("dark_path", "missing.npy")
        assert "the Dark (missing.npy not found)" in page._headers["processing"].detail
    finally:
        page.close_page()


def test_stitch_and_rsm_lay_out_one_corrections_group(qapp):
    """G8d (G8c unit 4c): Stitch's and RSM's Advanced Corrections rows are one shared group (``correction_values``):
    the same keys, labels, defaults, choices, file rows and tips, in one order. Stitch keeps its own Error model
    after Polarization and its stacked-arm sentence on Solid angle."""
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.gui.widgets.controls_panel import SubsectionCard

    def shown(page, card):  # each row in the card's order: its field's label, default, choices, browse, and tip
        layout, keys = card.body_layout, {row: key for key, row in page._rows.items()}
        rows = (layout.itemAt(index).widget() for index in range(layout.count()))
        return {keys[row]: (page._fields[keys[row]].label, page._fields[keys[row]].value,
                            page._fields[keys[row]].choices, page._fields[keys[row]].browse, row.toolTip())
                for row in rows}
    stitch, page = StitchPage(empty_host_services(_Status())), _page(qapp)
    try:
        cards = {card.title.text(): card for card in page.advanced.findChildren(SubsectionCard)}
        mine, theirs = shown(page, cards["Corrections"]), shown(stitch, stitch.corrections_card)
        assert list(theirs) == ["solid_angle", "polarization", "error_model", "dark_path", "flat_path",
                                "flat_cutoff"]
        assert theirs.pop("error_model")[:3] == ("Error model", "None", ("None", "Poisson"))
        stacked = " Keep on for a stacked-arm calibration with a sample offset: its frames sit at different distances."
        assert theirs["solid_angle"][4].endswith(stacked)
        theirs["solid_angle"] = (*theirs["solid_angle"][:4], theirs["solid_angle"][4][:-len(stacked)])
        assert list(mine) == list(theirs) and mine == theirs
        assert mine["solid_angle"][1:3] == ("On", ("On", "Off")) and mine["dark_path"][3] and mine["flat_path"][3]
        assert all(entry[4] for entry in mine.values())  # every row says what it does
    finally:
        stitch.shutdown()
        stitch.deleteLater()
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


def test_a_saved_calibration_shows_the_numeric_auto_ceiling_before_preview(qapp, tmp_path):
    """The real calibration exporter supplies the detector, never the window's unsaved defaults."""
    from xrd_tools.core.geometry.camera import DetectorCamera
    from xrd_tools.integrate.stitch_calibration import save_calibration_pair, pair_validation_angles

    _synthetic_source(tmp_path, full=True)
    page = _page(qapp)
    try:
        _with_source(page, tmp_path)
        page._threshold_edited(THRESHOLD_ENABLED, True)
        page._calibrate()
        dialog = page._calibration_dialog
        dialog.camera.set_value(DetectorCamera("z-", "x-"))
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData("manual"))
        for key, value in (("beam_detector", "Pilatus100k"), ("rough_distance", "1014.7173"),
                           ("beam_pixel", "97, 243"), ("rough_energy", "13.000007")):
            dialog._set_field_text(key, value)
        fit = dialog._rough_fit()
        target = tmp_path / "calibration" / "early.json"
        saved = save_calibration_pair(fit, target, validation_angles=pair_validation_angles(fit))
        assert saved.xu_path is not None, saved.xu_refusal
        dialog.close()

        def hints():
            return {edit.placeholderText() for edit in page.threshold_range.findChildren(QtWidgets.QLineEdit)}

        assert "Auto · the detector's saturation" in hints()  # unsaved window defaults do not leak
        page.set_field("geometry_path", "calibration/early.json")
        assert "Auto · 1048575 (saturation)" in hints()
        assert page._auto_threshold_max == 1048574.0
        assert page._threshold_state.threshold_max is None
        page.set_sources(page._source_paths)
        assert "Auto · 1048575 (saturation)" in hints()
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        # This fixture's uint16 ceiling is below the detector's counter limit:
        # Preview retains the existing dtype rule (None), replacing the early hint.
        assert page._owner.prepared.request.plan.corrections.saturation is None
        assert page._auto_threshold_max is None
        assert "Auto · the detector's saturation" in hints()
        page._threshold_edited(THRESHOLD_MAX, 10000.0)
        page.set_field("geometry_path", "calibration/missing.json")
        assert page._auto_threshold_max is None
        assert page._threshold_state.threshold_max == 10000.0
        assert "Auto · the detector's saturation" in hints()
        page.set_field("geometry_path", "calibration/early.json")
        assert page._auto_threshold_max == 1048574.0
        page.set_field("project_root", str(tmp_path / "elsewhere"))
        assert page._auto_threshold_max is None  # relative calibration is resolved against Project
        assert page._threshold_state.threshold_max == 10000.0
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


_NO_UB_CHOICES = "Enter one, or take it from a SPEC scan, or choose Q coordinates."


def test_the_no_ub_warning_names_the_scans_that_record_none(qapp, tmp_path):
    """G13 unit 2 (the maintainer, 2026-10-01: "will be better if it does name the scan"): the Source header's
    warning, and Preview's, name the members that record no UB. One RSM of both scans names both; once one records
    a UB, the other alone."""
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _own_ubs(tmp_path, (None, None))
        _with_source(page, tmp_path)  # [1-2]: one RSM of both scans
        both = "Scans 1 and 2 record no UB matrix. " + _NO_UB_CHOICES
        assert page._headers["source"].detail == both and not page._readiness[1]
        page.begin_preview()  # found again, and waits
        _wait(page, qapp, lambda: not page.active())
        assert label.full_text() == both and page._owner.prepared is None
        _own_ubs(tmp_path, ((1, 0, 0, 0, 1, 0, 0, 0, 1), None))
        page.resolve_source()
        assert page._headers["source"].detail == "Scan 2 records no UB matrix. " + _NO_UB_CHOICES
    finally:
        _close(page, qapp)


def test_the_no_ub_warning_names_three_then_counts_and_tells_files_and_entries_apart():
    """G13 unit 2: a scan is named as a selection names it; past three members the warning counts the rest; scans
    from several SPEC files carry their file's name; a NeXus stack is named by its file, with its entry when it is
    not ``entry``."""
    from types import SimpleNamespace
    from xdart.gui.tabs.rsm.page import no_ub_warning

    def scan(number, name="a.spec"):
        return SimpleNamespace(spec_path=f"/data/{name}", scan=number)

    def stack(name, entry="entry"):
        return SimpleNamespace(nexus_path=f"/data/{name}", entry=entry)

    for members, named in (
            ((scan("44"),), "Scan 44 records"),
            ((scan("44.1"), scan("45.2")), "Scans 44 and 45.2 record"),  # a selection's names: N, then N.2
            ((scan("41"), scan("42"), scan("43")), "Scans 41, 42 and 43 record"),
            (tuple(scan(str(number)) for number in range(41, 46)), "Scans 41, 42, 43 and 2 more record"),
            ((scan("44"), scan("44", "b.spec")), "Scans 44 of a.spec and 44 of b.spec record"),
            ((stack("member_noub.nxs"),), "member_noub.nxs records"),
            ((scan("44"), stack("s.nxs", "entry2")), "Scan 44 and s.nxs (entry2) record")):
        assert no_ub_warning(members) == f"{named} no UB matrix. {_NO_UB_CHOICES}"


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
        warning = "member_noub.nxs records no UB matrix. " + _NO_UB_CHOICES  # the member named (G13 unit 2)
        assert page.run_bar.readinessLabel.full_text() == warning and page._owner.prepared is None
        header = page._headers["source"]
        assert header.detail == warning and not header.ready and "⚠ no UB matrix" in header.text
        assert not page._readiness[1] and "a UB matrix" in page._readiness[0]
        assert not page.ub_card._collapsed  # the UB rows show
        page.begin_run()  # Run waits too
        assert not page.active() and page._owner.prepared is None
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        assert page._readiness[1] and page._headers["source"].ready and "no UB" not in page._headers["source"].detail
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
        assert "no UB" not in page._headers["source"].detail
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
        page.set_field("solid_angle", "Off")
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


def test_the_preview_names_the_calibrations_kind_from_its_provenance_outside_every_identity(qapp, tmp_path):
    """Item 7 (row C's finding 2): under "Geometry: the calibration on the Calibration row" the Preview says what
    kind of calibration the pair is, from its provenance in plain words: the calibration window's
    ``calibration_kind`` for a rough (direct-beam) pair, the calibrant frames of a ring fit, or that none is
    recorded. Display only: the three give the same summary, request, plan and output identities."""
    import json
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2, rsm_preflight_summary_lines
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        provenance = tmp_path / "calibration" / "rsm_canonical.xu.json.provenance.json"
        seen = {}
        for name, record in (("none", None),
                             ("rough", {"calibration_kind": "direct_beam_not_ring_fitted", "frames": []}),
                             ("ring", {"frames": [{"index": 0}, {"index": 3}, {"index": 7}], "rms_q_A^-1": 0.0021})):
            if record is not None:
                provenance.write_text(json.dumps(record))
            summary = prepare_rsm_tool_v2(page.build_form()).summary
            lines = rsm_preflight_summary_lines(summary)
            geometry = next(index for index, line in enumerate(lines) if line.startswith("Geometry: the calibration"))
            seen[name] = (lines[geometry + 1], (summary.fingerprint, summary.request_fingerprint,
                                                summary.plan_fingerprint, summary.output_fingerprint))
        assert seen["none"][0] == "  calibration: its kind is not recorded (no provenance beside the pair)"
        assert seen["rough"][0] == ("  calibration: direct-beam (rough): the distance and beam pixel, rotations "
                                    "held; not ring-fitted")
        assert seen["ring"][0] == "  calibration: ring-fitted on 3 calibrant frames (RMS |q| 0.0021 Å⁻¹)"
        assert seen["none"][1] == seen["rough"][1] == seen["ring"][1]
        page.begin_preview()  # the page's own Preview shows it
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        assert seen["ring"][0] in page.preview_text.toPlainText().splitlines()
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


def _panels_at(values, indices, *, projection=True):
    """``values``' six panels at ``indices``, as an independent RSMViewerModel draws them."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    model = RSMViewerModel()
    model.snapshot(values)
    return rsm_keyed_projection(values, model.snapshot(values, h_index=indices[0], k_index=indices[1],
                                                       l_index=indices[2], projection=projection)).panels


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
        want = np.subtract(_panels_at(values["B"], stepped, projection=False)[0].image,
                           _panels_at(values["A"], stepped, projection=False)[0].image)
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


def test_a_background_admitted_again_beside_another_result_leaves_that_result_drawn(qapp, monkeypatch):
    """G8d's self-review: the background's result admitted again while another is shown (a stale run's output,
    a late read: ``show=False``) releases the background, and the result shown is drawn again without it; the
    view's release had cleared its panels, which stayed blank until the next draw."""
    values = {"A": _grid_values(1.0), "A2": _grid_values(5.0), "B": _grid_values(3.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.background.click()
        page._add_result("/tmp/b_rsm.nexus", "B")
        assert page.view.keyed_projection.background_set
        page._add_result("/tmp/a_rsm.nexus", "A2", show=False)
        assert page._background.phase == "RELEASED"
        drawn = page.view.keyed_projection
        assert drawn is not None and not drawn.background_set and page.view.background.text() == "Set BG"
        own = _panels_at(values["B"], page.shown[1].state.axis_indices)
        for got, panel in zip(_drawn_arrays(page.view), own, strict=True):
            np.testing.assert_array_equal(got, panel.image if hasattr(panel, "image") else panel.intensity)
    finally:
        _close(page, qapp)


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


# AUD-9 stage 2b G9: the maintainer's two extra tools, "as extra tools that pop up separate windows to keep the
# scientific view clean for now"


class _Minimised(QtGui.QHideEvent):
    """The hide event a window gets when it is minimised: spontaneous (the window system's), not a close."""

    def spontaneous(self):
        return True


def test_the_3d_viewer_opens_on_the_result_shown_follows_it_and_closes_with_the_page(qapp, monkeypatch):
    """"A button to bring up a 3d viewer for the entire 3d RSM volume": the 3D Viewer button opens a non-modal
    window on the result shown (its volume shared, not copied), which follows the result shown; closing the page
    releases it."""
    values = {"A": _grid_values(1.0), "B": _grid_values(3.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page.view.page_tool("3d").click()  # nothing shown yet: the page says what the tool needs
        assert "3d" not in page._tool_windows
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("3d").click()
        _settled(page._tool_windows["3d"])
        window = page._tool_windows["3d"]
        assert window.isVisible() and not window.isModal()
        assert window.windowTitle() == "RSM 3D Viewer — a_rsm.nexus"
        assert np.shares_memory(window.view.projection.values, values["A"].intensity)
        assert window.view.projection.labels == tuple(values["A"].coordinate_frame.axis_labels)
        page._add_result("/tmp/b_rsm.nexus", "B")  # the window follows the result shown
        _settled(window)
        assert window.windowTitle() == "RSM 3D Viewer — b_rsm.nexus"
        assert np.shares_memory(window.view.projection.values, values["B"].intensity)
        page.view.page_tool("3d").click()  # a later press raises the same window
        _settled(page._tool_windows["3d"])
        assert page._tool_windows["3d"] is window
        window.hideEvent(_Minimised())  # minimised, not closed: the window keeps its volume
        assert np.shares_memory(window.view.projection.values, values["B"].intensity)
        window.close()  # closed by the user: its volume and RGBA go until it is shown again
        assert window.view.projection is None and window.view.rgba is None
        page.view.page_tool("3d").click()
        _settled(page._tool_windows["3d"])
        assert np.shares_memory(window.view.projection.values, values["B"].intensity)
    finally:
        _close_settled(page, qapp)
    assert window.view.projection is None and window.view.rgba is None and not page._tool_windows


def _type(editor, text):
    """Type ``text`` into a row's line editor as a user does: the edit commits on ``editingFinished``."""
    editor.setText(text)
    editor.setModified(True)
    editor.editingFinished.emit()


def _settled(window, seconds=20.0):
    """A tool window once what it was asked for is done (item 6: each computes on its own worker): its last plane
    or draw drawn or dropped as the event loop delivers it, and its worker idle."""
    from xdart.gui.tabs.rsm.tool_windows import RSMVolumeWindow
    worker = window.view if isinstance(window, RSMVolumeWindow) else window
    deadline = time.monotonic() + seconds
    while (worker.computing or worker._running is not None) and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.002)
    assert not worker.computing and worker._running is None
    return window


def _typed(window, editor, text):
    """``_type`` into one of a tool window's rows, then the window once its plane is drawn or refused."""
    _type(editor, text)
    _settled(window)


def _close_settled(page, qapp):
    """``_close`` once the page's tool windows have no plane or draw still computing (item 6: one computing makes
    the page's close PENDING until it returns; ``test_closing_the_page_while_a_tool_computes_is_pending...``)."""
    for window in page._tool_windows.values():
        _settled(window)
    _close(page, qapp)


def test_custom_slice_opens_on_hh_l_through_the_centre_follows_the_result_and_closes_with_the_page(
        qapp, monkeypatch):
    """"So say HH and l": the Custom Slice button opens a non-modal window on the result shown, cutting HH–L
    through the volume's centre (``custom_slice``'s defaults) on Int's view, its axes the coordinates along HH
    and L, saying the plane is interpolated; Int's Log draws it again; it follows the result shown, and closing
    the window or the page lets the volume go."""
    from xrd_tools.rsm.custom_slice import custom_slice, volume_centre
    values = {"A": _grid_values(1.0), "B": _grid_values(3.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page.view.page_tool("custom").click()  # nothing shown yet: the page says what the tool needs
        assert "custom" not in page._tool_windows
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        assert window.isVisible() and not window.isModal()
        assert window.windowTitle() == "RSM Custom Slice — a_rsm.nexus"
        assert window.rows["first"].current_value() == "1 1 0" and window.rows["second"].current_value() == "0 0 1"
        assert window.rows["first"].label.text() == "Direction 1 (h k l)"
        axes = tuple(axis for _name, axis in values["A"].axes)
        centre = volume_centre(axes)
        expected = custom_slice(values["A"].intensity, axes)  # HH–L through the centre
        drawn = window.view.keyed_projection
        assert drawn.share_axis and window.view.share_axis.isChecked()
        plane, along_hh, along_l = drawn.panels
        assert drawn.layout == ((plane.key,), (along_hh.key, along_l.key)) and not plane.navigation
        np.testing.assert_array_equal(plane.image, expected.values.T)
        assert (plane.x_axis.label, plane.y_axis.label) == ("HH (r.l.u.)", "L (r.l.u.)")
        np.testing.assert_allclose(plane.x_axis.values, expected.t + (centre[0] + centre[1]) / 2.0, rtol=0,
                                   atol=1e-12)
        np.testing.assert_allclose(plane.y_axis.values, expected.s + centre[2], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(along_hh.intensity, expected.projection_t)
        np.testing.assert_array_equal(along_l.intensity, expected.projection_s)
        assert "interpolated from the gridded volume" in window.note.text()
        assert window.view.keyed_projection.share_axis
        plane_box = window.view.keyed_panel(plane.key).pane.canvas.imageViewBox
        plane_box.setRange(xRange=(-0.2, 0.1), yRange=(-0.3, 0.4), padding=0)
        for key, expected_range in ((along_hh.key, (-0.2, 0.1)), (along_l.key, (-0.3, 0.4))):
            np.testing.assert_allclose(window.view.keyed_panel(key).plot.getViewBox().viewRange()[0], expected_range)
        window.view.share_axis.click()
        assert window.card.status.isHidden() and not window.card.valid_marker.isHidden()
        assert window._values is values["A"]  # the volume shared, not copied
        window.view.log_scale.click()  # Int's Log (and colour map) draw the plane again
        assert window.view.keyed_projection.log_scale
        page._add_result("/tmp/b_rsm.nexus", "B")  # the window follows the result shown
        _settled(window)
        assert window.windowTitle() == "RSM Custom Slice — b_rsm.nexus"
        np.testing.assert_array_equal(window.view.keyed_projection.panels[0].image,
                                      custom_slice(values["B"].intensity, axes).values.T)
        assert window.view.keyed_projection.log_scale
        page.view.page_tool("custom").click()  # a later press raises the same window
        _settled(page._tool_windows["custom"])
        assert page._tool_windows["custom"] is window
        window.hideEvent(_Minimised())  # minimised, not closed: the window keeps its plane
        assert window._values is values["B"] and window.custom is not None
        window.close()  # closed by the user: the volume and the plane go until it is shown again
        assert window._values is None and window.custom is None and window.view.keyed_projection is None
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        assert window._values is values["B"] and window.custom is not None
    finally:
        _close_settled(page, qapp)
    assert window._values is None and window.custom is None and not page._tool_windows


def test_an_axis_aligned_custom_slice_is_the_main_views_hk_slice(qapp, monkeypatch):
    """Directions 1 0 0 and 0 1 0 through the centre, one point per voxel: the plane, its axes and its
    orientation are the main view's H–K slice at that L."""
    from xrd_tools.session.rsm_viewer_model import RSM_HK
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        for name, text in (("first", "1 0 0"), ("second", "0 1 0"), ("first_points", "5"), ("second_points", "7")):
            _typed(window, window.rows[name].editor, text)
        plane = window.view.keyed_projection.panels[0]
        hk = next(panel for panel in _panels_at(values["A"], (0, 0, 4), projection=False) if panel.key == RSM_HK)
        assert plane.image.shape == hk.image.shape == (7, 5)
        # the viewer's axes are float32 (K = 1 + 5/6 is 1.8333334), the plane's coordinates float64
        np.testing.assert_allclose(plane.x_axis.values, hk.x_axis.values, rtol=0, atol=1e-6)
        np.testing.assert_allclose(plane.y_axis.values, hk.y_axis.values, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(plane.image, hk.image)
        assert (plane.x_axis.label, plane.y_axis.label) == ("H (r.l.u.)", "K (r.l.u.)")
        low, high = window.rows["first_range"].findChildren(QtWidgets.QLineEdit)
        _typed(window, low, "0.25")
        _typed(window, high, "0.75")  # a range is H itself, not an offset from the point
        ranged = window.view.keyed_projection.panels[0]
        np.testing.assert_allclose(ranged.x_axis.values, np.linspace(0.25, 0.75, 5), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(ranged.image[:, ::2], hk.image[:, 1:4])  # H 0.25, 0.5, 0.75: voxels 1–3
    finally:
        _close_settled(page, qapp)


def test_each_plane_custom_slice_refuses_is_named_in_its_card(qapp, monkeypatch):
    """Zero and parallel directions, an all-empty plane, unreadable text and a half-typed range are refused:
    the card's readiness and the view say why and nothing is drawn; a readable plane is drawn again."""
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        low, high = window.rows["first_range"].findChildren(QtWidgets.QLineEdit)
        cases = (
            (window.rows["first"].editor, "0 0 0", "1 1 0", "the first direction must be non-zero"),
            (window.rows["second"].editor, "2 2 0", "0 0 1", "the two directions are parallel"),
            (window.rows["through"].editor, "1 x 0", "", "Through needs three numbers"),
            (window.rows["first_points"].editor, "1", "", "Points along 1"),
            (low, "50", "", "Along 1 needs both bounds, or neither (Auto)"),
        )
        for editor, refused, readable, reason in cases:
            _typed(window, editor, refused)
            assert reason in window.card.status.text() and not window.card.status.isHidden(), reason
            assert window.card.valid_marker.isHidden() and window.custom is None
            assert window.view.keyed_projection is None and reason in window.view.status.text()
            _typed(window, editor, readable)
            assert window.custom is not None and window.card.status.isHidden(), reason
        _typed(window, low, "50")
        _typed(window, high, "60")  # HH from 50 to 60: the plane holds no data
        assert "the plane does not cross the volume's data" in window.card.status.text()
        assert window.custom is None and window.view.keyed_projection is None
    finally:
        _close_settled(page, qapp)


def test_custom_slice_in_q_coordinates_takes_qx_qy_qz_vectors(qapp, monkeypatch):
    """In the Q frame the directions and the point are Qx Qy Qz vectors, the axes in Å⁻¹."""
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
    from xrd_tools.rsm.custom_slice import custom_slice
    from xrd_tools.session.rsm_viewer_model import make_rsm_viewer_values
    names = ("qx", "qy", "qz")
    axes = tuple((name, np.linspace(index, index + 1.0, size))
                 for index, (name, size) in enumerate(zip(names, (5, 7, 9), strict=True)))
    values = {"Q": make_rsm_viewer_values(project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=axes, axis_units=tuple((name, "q_A^-1") for name in names),
        intensity=np.arange(5 * 7 * 9, dtype=np.float64).reshape(5, 7, 9) + 1.0, sigma=None, coverage=None,
        normalization=None))}
    assert values["Q"].coordinate_frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/q_rsm.nexus", "Q")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        assert [window.rows[name].label.text() for name in ("first", "second", "through")] == [
            "Direction 1 (Qx Qy Qz)", "Direction 2 (Qx Qy Qz)", "Through (Qx Qy Qz)"]
        plane = window.view.keyed_projection.panels[0]
        assert (plane.x_axis.label, plane.y_axis.label) == ("t·[1 1 0] (Å⁻¹)", "Qz (Å⁻¹)")
        expected = custom_slice(values["Q"].intensity, tuple(axis for _name, axis in values["Q"].axes),
                                symbols=("Qx", "Qy", "Qz"), unit="Å⁻¹")
        np.testing.assert_array_equal(plane.image, expected.values.T)
    finally:
        _close_settled(page, qapp)


_HKL_CAVEAT = (" In HKL the normal and the thickness are r.l.u. ones, H, K and L taken as orthonormal: for a "
               "non-orthogonal lattice they are not the reciprocal-space normal and distance.")


def test_custom_slice_thickness_averages_a_slab_and_the_note_names_it(qapp, monkeypatch):
    """G11 (the maintainer, 2026-10-01: "Do you want the Thickness option? Yes"): a shared Thickness row, 0 by
    default, in the volume's units. A thickness draws ``custom_slice``'s slab (its planes' finite mean, and the
    slab's projections); the note names the planes and, in HKL, that the normal and the thickness are r.l.u.
    ones; 0 (or blank) is the plane alone again. In Q the note has no HKL caveat."""
    from xdart.gui.widgets.controls_panel import FormRow
    from xrd_tools.rsm.custom_slice import custom_slice
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        row = window.rows["thickness"]
        assert type(row) is FormRow and row.current_value() == "0"
        assert row.label.text() == "Thickness (r.l.u.)"
        plane = window.custom
        assert plane.thickness == 0.0 and "averaged" not in window.note.text() and "In HKL" not in window.note.text()
        axes = tuple(axis for _name, axis in values["A"].axes)
        expected = custom_slice(values["A"].intensity, axes, thickness=0.5)
        _typed(window, row.editor, "0.5")
        slab, drawn = window.custom, window.view.keyed_projection
        assert (slab.thickness, slab.planes) == (0.5, expected.planes) and expected.planes > 3
        np.testing.assert_array_equal(drawn.panels[0].image, expected.values.T)
        np.testing.assert_array_equal(drawn.panels[1].intensity, expected.projection_t)
        np.testing.assert_array_equal(drawn.panels[2].intensity, expected.projection_s)
        assert not np.array_equal(expected.values, plane.values, equal_nan=True)
        assert window.note.text() == (f"The plane is {plane.note}; averaged over {expected.planes} planes, ±0.25 "
                                      "along [1 −1 0]." + _HKL_CAVEAT)
        assert window.card.status.isHidden() and not window.card.valid_marker.isHidden()
        for no_thickness in ("0", ""):
            _typed(window, row.editor, no_thickness)
            assert window.custom.thickness == 0.0 and window.note.text() == f"The plane is {plane.note}."
            np.testing.assert_array_equal(window.view.keyed_projection.panels[0].image, plane.values.T)
    finally:
        _close_settled(page, qapp)
    names = ("qx", "qy", "qz")
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
    from xrd_tools.session.rsm_viewer_model import make_rsm_viewer_values
    q_axes = tuple((name, np.linspace(index, index + 1.0, size))
                   for index, (name, size) in enumerate(zip(names, (5, 7, 9), strict=True)))
    values["Q"] = make_rsm_viewer_values(project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=q_axes, axis_units=tuple((name, "q_A^-1") for name in names),
        intensity=np.arange(5 * 7 * 9, dtype=np.float64).reshape(5, 7, 9) + 1.0, sigma=None, coverage=None,
        normalization=None))
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/q_rsm.nexus", "Q")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        assert window.rows["thickness"].label.text() == "Thickness (Å⁻¹)"
        _typed(window, window.rows["thickness"].editor, "0.5")
        assert window.custom.planes > 3 and window.note.text().endswith(" along [1 −1 0].")
        assert "In HKL" not in window.note.text()
    finally:
        _close_settled(page, qapp)


def test_a_custom_slice_thickness_that_is_no_number_0_or_more_is_named_in_its_card(qapp, monkeypatch):
    """The Thickness row's readiness: a negative, non-finite or unreadable thickness is refused in the card and
    the view, nothing drawn, the typed text kept; a readable one is drawn again."""
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        _settled(page._tool_windows["custom"])
        window = page._tool_windows["custom"]
        editor = window.rows["thickness"].editor
        reason = "Thickness (r.l.u.): a number, 0 or more"
        for refused in ("-0.1", "thick", "nan", "inf", "-inf"):
            _typed(window, editor, refused)
            assert window.card.status.text() == reason and not window.card.status.isHidden(), refused
            assert window.card.valid_marker.isHidden() and window.custom is None and window.note.text() == ""
            assert window.view.keyed_projection is None and reason in window.view.status.text()
            assert editor.text() == refused
            _typed(window, editor, "0.2")
            assert window.custom is not None and window.custom.thickness == 0.2 and window.card.status.isHidden()
    finally:
        _close_settled(page, qapp)


# AUD-9 stage 2b item 6 (the maintainer: "yes go ahead with both"): Custom Slice cuts on its window's worker, the
# latest plane wins; the page's close waits for it


def _held_cuts(monkeypatch):
    """Custom Slice's real cut, each result held before it returns until ``release`` is set (the gate of
    test_volume_view); the first plane is let through."""
    from tests.xdart.scattering.test_volume_view import _Held
    from xdart.gui.tabs.rsm import tool_windows
    held = _Held(tool_windows._cut_plane)
    monkeypatch.setattr(tool_windows, "_cut_plane", held)
    held.release.set()
    return held


def _hold(held):
    held.release.clear()
    held.computed.clear()


def test_only_the_latest_plane_is_drawn_and_a_stale_one_is_let_go(qapp, monkeypatch):
    """While a plane is cut the window keeps the plane drawn and says it computes; a plane finished after a later
    one was asked for is dropped and let go; the plane drawn is the latest, custom_slice's to the byte."""
    import gc
    from xrd_tools.rsm.custom_slice import custom_slice
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    held = _held_cuts(monkeypatch)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        window = _settled(page._tool_windows["custom"])
        first, drawn = window.custom, window.view.keyed_projection
        _hold(held)
        _type(window.rows["first_points"].editor, "5")  # plane B, held on the worker once cut
        assert held.computed.wait(20)
        assert window.computing and window.custom is first and window.view.keyed_projection is drawn
        assert window.card.status.text() == "Computing the plane…" and not window.card.valid_marker.isHidden()
        _type(window.rows["second_points"].editor, "6")  # plane C, asked for while B is on the worker
        assert window._running[1].points == (5, 9) and window._wanted.points == (5, 6)
        held.release.set()
        _settled(window)
        expected = custom_slice(values["A"].intensity, tuple(axis for _name, axis in values["A"].axes),
                                points=(5, 6))
        assert window.custom.values.tobytes() == expected.values.tobytes()
        np.testing.assert_array_equal(window.view.keyed_projection.panels[0].image, expected.values.T)
        assert window.card.status.isHidden() and len(held.results) == 3
        gc.collect()
        assert held.results[1]() is None  # B's plane, dropped
    finally:
        held.release.set()
        _close_settled(page, qapp)


def test_closing_custom_slice_while_it_cuts_drops_the_plane_and_lets_it_go(qapp, monkeypatch):
    """Closed (not minimised) while a plane is cut: nothing is drawn when it returns and it is let go; shown again,
    the window cuts what its rows say."""
    import gc
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    held = _held_cuts(monkeypatch)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        window = _settled(page._tool_windows["custom"])
        _hold(held)
        _type(window.rows["first_points"].editor, "5")
        assert held.computed.wait(20)
        window.close()
        assert window.custom is None and not window.computing and window.view.keyed_projection is None
        held.release.set()
        _settled(window)
        assert window.custom is None and window._values is None and window.view.keyed_projection is None
        gc.collect()
        assert held.results[1]() is None
        page.view.page_tool("custom").click()
        _settled(window)
        assert window.custom.values.shape == (5, 9)
    finally:
        held.release.set()
        _close_settled(page, qapp)


def test_closing_the_page_while_a_tool_computes_is_pending_until_it_returns(qapp, monkeypatch):
    """The page's close goes through the tool windows' workers as through its own owners: PENDING while a plane
    is cut (the host asks again), CLEAN once it returned, the plane never drawn and let go."""
    import gc
    values = {"A": _grid_values(1.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    held = _held_cuts(monkeypatch)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.page_tool("custom").click()
        window = _settled(page._tool_windows["custom"])
        first = window.custom
        _hold(held)
        _type(window.rows["first_points"].editor, "5")
        assert held.computed.wait(20)
        assert page.close_page().status is PageCleanup.PENDING
        assert window.custom is first
        held.release.set()
        deadline = time.monotonic() + 20
        receipt = page.close_page()
        while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
            time.sleep(0.01)
            receipt = page.close_page()
        assert receipt.status is PageCleanup.CLEAN and not page._tool_windows
        assert window.custom is None and window._running is None
        gc.collect()
        assert held.results[1]() is None
    finally:
        held.release.set()
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("message", (
    "analysis artifact published; verification remains pending",
    "analysis artifact document close remains pending",
))
def test_a_pending_finalization_notice_keeps_the_actual_publication_state(qapp, message):
    """A saving left pending names what actually happened (the owner's finalization message, published or not)
    and offers Retry without regridding. Moved from the classic RSM dialog's row when it was retired (G10)."""
    status = _Status()
    page = RSMPage(empty_host_services(status), owner=_Owner())
    try:
        page._accept(RSMOwnerUpdate(
            OperationIdentity(1, object()), RSMOwnerAction.RUN, terminal_status=OperationTerminalStatus.RETURNED,
            outcome=RSMOwnerOutcome(RSMOwnerOutcomeKind.CLEANUP_PENDING, finalization_message=message)))
        shown = page.run_bar.readinessLabel.full_text()
        assert message in shown and "Retry" in shown
        assert any(message in text for text, _timeout in status.messages)
    finally:
        _close(page, qapp)


def test_an_edit_after_preview_drops_the_prepared_form(qapp, tmp_path):
    """The classic RSM dialog bound Run to the exact form its Preview prepared (moved, G10). On the page any edit
    drops that prepared form's fingerprint, so a Run previews the edited inputs again instead of running the
    stale Preview (``_begin_action`` reuses a Preview only for the same fingerprint); a refresh with no edit
    keeps it."""
    page = _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        form = page.build_form()
        page._prepared_fingerprint = form.fingerprint
        page._sync_actions()
        assert page._prepared_fingerprint == form.fingerprint
        page.set_field("grid", "Full 200×200×200")
        assert page._prepared_fingerprint is None
        assert page.build_form().fingerprint != form.fingerprint
    finally:
        _close(page, qapp)


def test_a_presentation_failure_in_poll_is_contained_with_its_stable_code(qapp, monkeypatch):
    """A failure while the page presents an owner update is caught in ``_poll``: the page names it with its
    stable code and the error, and its timer stops once the owner is idle. Moved from the classic RSM dialog's
    row when it was retired (G10)."""
    page = _page(qapp, owner=_Owner())
    monkeypatch.setattr(page._owner, "poll", lambda: object())
    monkeypatch.setattr(page, "_accept", lambda _update: (_ for _ in ()).throw(RuntimeError("paint fault")))
    try:
        page._timer.start()
        page._poll()
        shown = page.run_bar.readinessLabel.full_text()
        assert "RSM_VIEW_PRESENTATION_FAILED" in shown and "paint fault" in shown
        assert not page._timer.isActive()
    finally:
        _close(page, qapp)
