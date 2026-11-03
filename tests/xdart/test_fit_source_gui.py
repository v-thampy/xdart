"""NEXT-128-WINDOW-POLISH item 11: the fit windows' Source card and their Batch over a NeXus result's stored rows.

The maintainer, 2026-11-03: "I think it will be good to have an option to choose a separate source than the nexus file
selected in the main window in the fit windows. These should probably be the first thing in the left panel." His
answers: "Yes, in 1.2.8 with Batch (Recommended)" and "Stored rows; fit the substrate (Recommended)". Rows S1-S7 of
DESIGN-FIT-SOURCE.md, frozen before the product change on the real Int page: its router, the shared fit host's worker
and the real reader (no fake on those seams). The displayed frame is supplied as the p37 rows supply it.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

import xdart.gui.tabs.scattering.analysis_mount as mount
from tests.core.test_int_fit_source import Q, _peak, _result, _stored
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_values import AxisProjection, ScientificProjection, TraceProjection


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _drain(qapp, host):
    deadline = time.monotonic() + 30
    while host.computing and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not host.computing
    qapp.processEvents()


def _window(page, kind):
    page._open_analysis_mount(kind)
    dialog = getattr(page, f"_{kind}_dialog")
    return dialog, page._fit_hosts[kind]


def _choose_nexus(dialog, path, frame, mode="default"):
    row = dialog.source_row
    row._file_chooser = lambda: str(path)
    row._ask_frame = lambda _path, _frame, _mode: (frame, mode)
    row.choose_btn.click()


def _displayed(path, label, *, unit="q_A^-1", wavelength=1.0):
    """The trace Int displays: frame ``label`` of the file at ``path``, on ``unit`` (Int's 2θ view of q rows converts
    with ``convert_radial_axis`` at the file's wavelength)."""
    from xrd_tools.core.containers import convert_radial_axis
    frame = DisplayFrameKey(RunIdentity(2, "run-fingerprint"), "sample", str(path), label, 9)
    x = _stored(Q) if unit == "q_A^-1" else convert_radial_axis(_stored(Q), "q_A^-1", unit, wavelength)
    projection = ScientificProjection(traces=(TraceProjection(
        frame, AxisProjection(np.asarray(x, dtype="<f8"), "Q" if unit == "q_A^-1" else "2θ", unit),
        np.asarray(_stored(_peak(2.0 + 0.1 * label)), dtype="<f8"), "sample_2", 1700000000.25),))
    return mount.displayed_trace_input(projection, frame), SimpleNamespace(frame=frame)


def _single(snapshot, values):
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
    return run_displayed_peak_fit(DisplayedPeakFitPlan(snapshot.trace_input(), **values))


# ---- S1 the card ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_s1_ints_fit_windows_open_with_a_source_card_first_following_ints_display(qapp, kind):
    page = _page()
    try:
        dialog, _host = _window(page, kind)
        from xdart.gui.widgets.controls_panel import SectionCard
        assert isinstance(dialog.source_card, SectionCard)
        assert dialog.split._cards.itemAt(0).widget() is dialog.source_card
        assert dialog.source_card.isAncestorOf(dialog.source_row) and dialog.fit_source() is None
        row = dialog.source_row
        assert row.status.text().startswith("Int's display")
        for control in (row.follow_btn, row.choose_btn, row.status, dialog.source_card):
            assert control.toolTip().strip(), control
        for form in (row.frame_row, row.mode_row, row.unit_row):
            assert form.toolTip().strip() and form.editor.toolTip().strip(), form
        buttons = row._buttons
        assert buttons.itemAt(buttons.count() - 1).widget() is row.help_button  # the "?" rightmost
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_s1_the_other_hosts_windows_have_no_source_card(qapp, kind):
    from tests.xdart.scattering.test_snapshot_fit_host import sample
    from xdart.gui.analysis.snapshot_fit_host import SnapshotFitHost
    host = SnapshotFitHost(kind, current=lambda: sample(), scope=lambda: "scope")
    try:
        assert host.dialog.source_card is None and host.dialog.source_row is None and host.dialog.fit_source() is None
    finally:
        host.dialog.close()
        while host.shutdown().status.value == "pending":
            qapp.processEvents()


# ---- S2 a file source's Reload and Fit ---------------------------------------------------------------------------

def test_s2_a_nexus_source_reloads_and_fits_its_frame_on_the_stored_axis(qapp, tmp_path):
    from xrd_tools.analysis.background_template import BackgroundTemplateSource
    from xrd_tools.analysis.fit_sources import int_frame_snapshot
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 3)
        assert dialog.fit_source() == BackgroundTemplateSource("nexus", str(path), frame=3, mode="default")
        assert dialog.source_row.status.text().startswith("sample_int1d.nexus, frame 3")
        dialog.refresh_btn.click()
        _drain(qapp, host)
        assert np.array_equal(dialog._x, _stored(Q)) and np.array_equal(dialog._y, _stored(_peak(2.3)))
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert page._analysis_identity is None  # Int's display slot is not asked
        outcome = host.outcomes[-1]
        single = _single(int_frame_snapshot(path, 3, mode="default"), dialog.vnext_fit_values())
        assert outcome.ok and outcome.parameter_values == dict(zip(single.parameter_names, single.parameter_values))
        editor = dialog.source_row.frame_row.editor
        editor.setText("9")
        editor.editingFinished.emit()  # the user types another frame: it is read at once and refused by name
        _drain(qapp, host)
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert "FIT_SOURCE_FRAME_MISSING" in dialog.status.text() + dialog.status.toolTip()
    finally:
        _close_page(page, qapp)


def test_s2_int_display_returns_fit_and_reload_to_the_page(qapp, tmp_path, monkeypatch):
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 1)
        dialog.source_row.follow_btn.click()
        assert dialog.fit_source() is None and dialog.source_row.status.text().startswith("Int's display")
        trace, anchor = _displayed(path, 2)
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (trace, anchor))
        monkeypatch.setattr(mount, "display_anchor_matches", lambda *args: True)
        dialog.refresh_btn.click()
        assert np.array_equal(dialog._y, _stored(_peak(2.2))) and not host.computing  # the page's own Reload
        dialog.fit_btn.click()
        assert page._analysis_identity is not None and not host.computing  # the page's slot fits the display
    finally:
        _close_page(page, qapp)


# ---- S3 a file source's Batch ------------------------------------------------------------------------------------

def test_s3_batch_fits_every_frame_of_a_nexus_source_and_shows_the_last(qapp, tmp_path):
    from xrd_tools.analysis.fit_sources import int_frame_snapshot
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 1)
        assert dialog.batch_btn.isEnabled() and "every frame" in dialog.batch_btn.toolTip()
        dialog.batch_btn.click()
        _drain(qapp, host)
        labels, xs, columns = dialog.trend.table()
        assert len(labels) == 5 and xs == (0.0, 1.0, 2.0, 3.0, 4.0)
        values = dialog.vnext_fit_values()
        for frame, center in zip(range(5), columns["center_0"]):
            single = _single(int_frame_snapshot(path, frame, mode="default"), values)
            assert center == dict(zip(single.parameter_names, single.parameter_values))["p0_center"]
        assert np.array_equal(dialog._y, _stored(_peak(2.4)))  # the last frame is shown
    finally:
        _close_page(page, qapp)


# ---- S4 Int's display's Batch ------------------------------------------------------------------------------------

def test_s4_ints_display_batch_fits_the_displayed_files_frames_in_the_displayed_unit(qapp, tmp_path, monkeypatch):
    """With no norm channel, Set BG or χ slice, each frame's input is Int's displayed trace of that frame: here a 2θ
    view of q rows, converted with Int's function at the file's wavelength."""
    from xrd_tools.analysis.fit_sources import int_frame_snapshot
    path = _result(tmp_path / "sample_int1d.nexus")
    trace, anchor = _displayed(path, 2, unit="2th_deg")
    monkeypatch.setattr(mount, "displayed_mode_1d", lambda *_args: "default")
    page = _page()
    try:
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (trace, anchor))
        dialog, host = _window(page, "peak")
        assert dialog.batch_btn.isEnabled(), dialog.batch_btn.toolTip()
        tip = dialog.batch_btn.toolTip()
        assert "sample_int1d.nexus" in tip and "stored" in tip and "display-only" in tip
        dialog.batch_btn.click()
        _drain(qapp, host)
        labels, xs, columns = dialog.trend.table()
        assert xs == (0.0, 1.0, 2.0, 3.0, 4.0)
        values = dialog.vnext_fit_values()
        for frame, center in zip(range(5), columns["center_0"]):
            snapshot = int_frame_snapshot(path, frame, mode="default", unit="2th_deg")
            if frame == 2:
                assert np.array_equal(snapshot.axes[0].values, trace.axis)  # the displayed frame's own trace
                assert np.array_equal(snapshot.values, trace.intensity)
            single = _single(snapshot, values)
            assert center == dict(zip(single.parameter_names, single.parameter_values))["p0_center"]
        assert dialog._x_unit in {"2th_deg", "2θ (°)"} or host.snapshot.axes[0].unit == "2th_deg"
    finally:
        _close_page(page, qapp)


def test_s4_batch_waits_with_its_reason_while_int_displays_no_nexus_frame(qapp):
    page = _page()
    try:
        dialog, _host = _window(page, "peak")  # nothing displayed
        assert not dialog.batch_btn.isEnabled() and dialog.batch_btn.toolTip().strip()
    finally:
        _close_page(page, qapp)


# ---- S5 Phase Analysis ---------------------------------------------------------------------------------------------

def test_s5_phase_analysis_reads_a_nexus_source_at_its_wavelength_and_refuses_an_xye(qapp, tmp_path):
    path = _result(tmp_path / "sample_int1d.nexus", wavelength=0.7293)
    page = _page()
    try:
        dialog, host = _window(page, "phase")
        _choose_nexus(dialog, path, 2)
        dialog.refresh_btn.click()
        _drain(qapp, host)
        assert host.snapshot is not None and host.snapshot.wavelength_angstrom == pytest.approx(0.7293, rel=1e-12)
        xye = tmp_path / "pattern.xye"
        xye.write_text("".join(f"{q:.6f} {y:.6f}\n" for q, y in zip(Q, _peak(2.5))), encoding="utf-8")
        row = dialog.source_row
        before = dialog.fit_source()
        row._file_chooser = lambda: str(xye)
        row._ask_unit = lambda *_args: pytest.fail("Phase Analysis asks no .xye unit: it refuses the file")
        row.choose_btn.click()
        assert dialog.fit_source() == before
        assert row.message_box is not None and "wavelength" in (row.message_box.text()
                                                                 + row.message_box.informativeText())
    finally:
        _close_page(page, qapp)


# ---- S6 notebook export names the source ---------------------------------------------------------------------------

def test_s6_a_nexus_source_exports_a_notebook_naming_its_file_and_frames(qapp, tmp_path):
    from xrd_tools.notebooks import run_recipe
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 3)
        _drain(qapp, host)  # the chosen frame is shown at once
        dialog.fit_btn.click()
        _drain(qapp, host)
        fitted = host.outcomes[-1]
        destination = tmp_path / "fit.ipynb"
        dialog.export_actions.export(destination)
        _drain(qapp, host)
        assert destination.is_file(), dialog.status.text()
        assert not tuple(tmp_path.glob("fit.ipynb*.input.npz"))  # the file is named, not captured
        config = next(cell["source"] for cell in json.loads(destination.read_text())["cells"]
                      if cell["cell_type"] == "code" and "recipe = recipe_from_json" in cell["source"])
        namespace = {}
        exec(config, namespace)
        recipe = namespace["recipe"]
        assert recipe.input.reader == "int_1d" and dict(recipe.input.selection)["frame"] == 3
        assert len(recipe.batch) == 5 and {item.reader for item in recipe.batch} == {"int_1d"}
        outcome = run_recipe(recipe, bundle_root=tmp_path, data_root=tmp_path)
        assert outcome.ok and outcome.parameter_values == fitted.parameter_values
    finally:
        _close_page(page, qapp)


# ---- S7 the gates --------------------------------------------------------------------------------------------------

def test_s7_an_xye_source_cannot_batch_and_a_running_batch_holds_ints_fits(qapp, tmp_path):
    from xdart.gui.tabs.scattering.analysis_mount import analysis_start_allowed
    xye = tmp_path / "pattern.xye"
    xye.write_text("".join(f"{q:.6f} {y:.6f}\n" for q, y in zip(Q, _peak(2.5))), encoding="utf-8")
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        row = dialog.source_row
        row._file_chooser = lambda: str(xye)
        row._ask_unit = lambda *_args: "q_A^-1"
        row.choose_btn.click()
        assert dialog.fit_source().kind == "xye"
        assert not dialog.batch_btn.isEnabled() and "one curve" in dialog.batch_btn.toolTip()
        _choose_nexus(dialog, path, 0)
        _drain(qapp, host)  # the chosen frame is shown at once
        assert dialog.batch_btn.isEnabled()
        allowed = analysis_start_allowed(page)
        dialog.batch_btn.click()
        assert host.computing and allowed and not analysis_start_allowed(page)
        assert dialog.busy_probe()
        _drain(qapp, host)
        assert analysis_start_allowed(page) == allowed
    finally:
        _close_page(page, qapp)


def test_s6_an_xye_source_exports_its_one_curve_as_a_captured_input(qapp, tmp_path):
    """Found in the self-review of 0453a194: an .xye source has no frames to batch, so its export asked for a batch,
    was refused, and wrote nothing. Its notebook carries the one curve as a captured sidecar."""
    xye = tmp_path / "pattern.xye"
    xye.write_text("".join(f"{q:.6f} {y:.6f}\n" for q, y in zip(Q, _peak(2.5))), encoding="utf-8")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        row = dialog.source_row
        row._file_chooser = lambda: str(xye)
        row._ask_unit = lambda *_args: "q_A^-1"
        row.choose_btn.click()
        _drain(qapp, host)
        destination = tmp_path / "fit.ipynb"
        dialog.export_actions.export(destination)
        _drain(qapp, host)
        assert destination.is_file(), dialog.status.text()
        assert len(tuple(tmp_path.glob("fit.ipynb*.input.npz"))) == 1
    finally:
        _close_page(page, qapp)


_EIGER = (Path(os.environ.get("XDART_TEST_DATA", "/nonexistent")) / "xdart_processed_data"
          / "eiger_S069Ta_redo_eta2p0_1_scan001_int1d.nexus")


@pytest.mark.skipif(not _EIGER.is_file(), reason="needs XDART_TEST_DATA (xdart_processed_data/eiger_..._int1d.nexus)")
@pytest.mark.parametrize("axis", ("Q", "2theta"))
def test_s4_a_real_browse_display_batches_on_its_displayed_unit(qapp, tmp_path, monkeypatch, _xdart_qt_harness, axis):
    """The advisor's pre-push check of the display-source Batch on Int's real Browse route (no fake): a real result
    shown as Q or as 2θ gives Batch a canonical displayed unit, the displayed frame's batch input is the displayed
    trace itself (no norm channel, Set BG or χ slice), and the Batch state's re-evaluation after a render is cheap
    (its median time is printed)."""
    import statistics
    from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    from xrd_tools.analysis.fit_sources import int_frame_snapshot
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        browse = page._context_controller.begin_browse(str(_EIGER), source_root=str(_EIGER.parent))
        outcome = _wait(page._context_controller.poll_browse, timeout=60)
        assert outcome.request is browse and outcome.status is BrowseLoadStatus.READY, outcome
        page.resize(1200, 800)
        page.show()
        page._refresh_shell()
        if axis == "2theta":
            page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "2theta"))
            page._refresh_shell()
        wanted = "q_A^-1" if axis == "Q" else "2th_deg"

        def painted():
            page._drain_executor()
            value = page._last_scientific_projection
            return value if value is not None and value.traces and value.traces[0].axis.unit == wanted else None
        _wait(painted, timeout=60)
        dialog, _host = _window(page, "peak")
        trace, _anchor = page._capture_analysis_trace("peak")
        assert trace is not None and trace.axis_unit == wanted
        assert dialog.batch_btn.isEnabled(), dialog.batch_btn.toolTip()
        assert _EIGER.name in dialog.batch_btn.toolTip()
        frame = int(trace.receipt.frame_label)
        snapshot = int_frame_snapshot(_EIGER, frame, mode="default", unit=trace.axis_unit)
        np.testing.assert_allclose(snapshot.axes[0].values, trace.axis, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(snapshot.values, trace.intensity)
        times = []
        for _ in range(15):
            start = time.perf_counter()
            page._sync_fit_batch("peak")
            times.append(time.perf_counter() - start)
        print(f"S4 real {axis}: unit {trace.axis_unit}, frame {frame}, Batch re-evaluation median "
              f"{statistics.median(times) * 1e3:.2f} ms (max {max(times) * 1e3:.2f} ms)")
    finally:
        _close_page(page, qapp)
