"""Stitch keeps the results it shows in memory, and the most recently shown others within a byte budget
(STITCH-2D-20260930 C12; Codex general audit AUD-5, narrowed by architecture invariant 12: a processed file can
always be read or stitched again; maintainer, 2026-09-30: "can we hold more than one, say abot 20, or even 48
so browsing is smoother?"). Measured on LaB6 scan 14 (287 frames): a Stitch 2D result holds 4.65 MB, a Stitch
1D 0.32 MB. The others are listed by name and reloaded from their files when shown again. Rows about reading
back set the budget to 0, so only the shown results stay.
"""
import pytest
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


@pytest.fixture
def no_recent(monkeypatch):
    from xdart.gui.tabs.stitch import page as module
    monkeypatch.setattr(module, "_RECENT_BYTES", 0)


def test_only_the_shown_results_stay_in_memory_and_the_others_reload(qapp, tmp_path, no_recent):
    page = StitchPage(empty_host_services(_Status()))
    select = QtCore.QItemSelectionModel.SelectionFlag
    role = QtCore.Qt.ItemDataRole.UserRole
    try:
        page.load_form(_native_form(tmp_path, "multigeometry"))  # three outputs
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        paths = [page.results.item(row).data(role) for row in range(page.results.count())]
        assert len(paths) == 3 and all(Path(path).is_file() for path in paths)
        # Single: the current result only (the last one stitched)
        assert set(page._payloads) == {paths[-1]}
        page.results.setCurrentRow(0)
        _wait(page, qapp)
        assert set(page._payloads) == {paths[0]} and page.frames.count() == 3
        # Overlay: the selected results
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page.results.setCurrentRow(2, select.ClearAndSelect)
        page.results.setCurrentRow(0, select.Select)
        _wait(page, qapp)
        assert set(page._payloads) == {paths[0], paths[2]}
        assert len(page.curve.listDataItems()) == 2
        # a result whose file is gone stays listed, with the reason, and holds nothing
        Path(paths[1]).unlink()
        page.results.setCurrentRow(1, select.ClearAndSelect)
        _wait(page, qapp)
        assert page.results.count() == 3 and paths[1] not in page._payloads
        status = page.run_bar.readinessLabel.full_text()
        assert Path(paths[1]).name in status and "No such file" in status, status
        assert not page.run_bar.readinessDot.property("ready")
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_a_pin_survives_selecting_another_result(qapp, tmp_path):
    """Codex terminal review F1: a pinned slice's result stays in memory when another result is selected, and
    its pinned trace stays drawn (it had vanished: the pinned result's map never reached the projection)."""
    command = lambda kind, value=None: page._view_command(ShellCommand(kind, value))  # noqa: E731
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_native_form(tmp_path, "multigeometry"), mode="2d", npt_rad_2d=48, npt_azim_2d=32))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page.results.count() == 3
        command(ShellCommandKind.SET_PLOT_MODE, "Overlay")
        command(ShellCommandKind.SET_SLICE_ENABLED, True)
        # inside the map: χ Auto spans the frames' own χ (here about -157 to -80°), as Int 2D's
        command(ShellCommandKind.SET_SLICE_CENTER, -120.0)
        command(ShellCommandKind.PIN_SLICE)
        (pinned,) = page._last_map_state.pinned_traces
        pinned_path = page.results.currentItem().data(QtCore.Qt.ItemDataRole.UserRole)
        page.results.setCurrentRow(0, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        _wait(page, qapp)
        assert pinned_path in page._payloads and len(page._view.slice_pins) == 1
        (kept,) = page._last_map_state.pinned_traces
        assert kept.trace.frame is pinned.trace.frame
        assert page._last_map_state.traces and all(
            trace.frame is not pinned.trace.frame for trace in page._last_map_state.traces)  # a pin, not selected
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_the_chosen_gi_axis_survives_reading_a_result_back(qapp, tmp_path, no_recent):
    """Codex terminal review F2: selecting a result that is being read back keeps the plot axis the user chose
    (it was reset to Q while nothing was loaded); the result shows on it once back."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        form = _gi_form(tmp_path, "multigeometry", StitchGI(
            ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE))
        page.load_form(replace(form, mode="2d", npt_rad_2d=48, npt_azim_2d=32,
                               source_selection=replace(form.source_selection, scans="1,2")))
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page.results.count() == 2 and len(page._payloads) == 1
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "q_oop"))
        page.results.setCurrentRow(0)
        assert page._view.plot_axis == "q_oop"  # while it is read back
        _wait(page, qapp)
        assert page._view.plot_axis == "q_oop" and page._last_map_state.plot_axis == "q_oop"
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_a_map_without_the_chosen_cut_shows_i_of_q_and_keeps_the_choice(qapp, tmp_path):
    """With a GI Stitch 2D's cut chosen, a plain Stitch 2D shows its I(q) (the axis falls back to Q for it),
    and the GI result shows the cut again: the choice is the user's, as Codex's F2 asks of a read-back."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        form = replace(_gi_form(tmp_path, "multigeometry", StitchGI(
            ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)),
            mode="2d", npt_rad_2d=48, npt_azim_2d=32)
        page.load_form(form)
        page.run_bar.startButton.click()  # the GI Stitch 2D
        _wait(page, qapp)
        page.load_form(replace(form, gi=None, output_family="plain"))
        page.run_bar.startButton.click()  # a plain Stitch 2D beside it
        _wait(page, qapp)
        assert page.results.count() == 2
        page.results.setCurrentRow(0)
        _wait(page, qapp)
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "q_oop"))
        gi_payload = page._current_payload()
        assert gi_payload.gi_map is not None and page._last_map_state.plot_axis == "q_oop"
        page.results.setCurrentRow(1)
        _wait(page, qapp)
        plain = page._current_payload()
        assert plain.gi_map is None and page._view.plot_axis == "q_oop"
        (trace,) = page._last_map_state.traces
        np.testing.assert_array_equal(trace.intensity, plain.profile.intensity)
        page.results.setCurrentRow(0)
        _wait(page, qapp)
        (trace,) = page._last_map_state.traces
        np.testing.assert_array_equal(trace.intensity, dict(page._current_payload().gi)["q_oop"].intensity)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_recently_shown_results_stay_within_the_budget_least_recent_first(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.stitch import page as module
    page = StitchPage(empty_host_services(_Status()))
    role = QtCore.Qt.ItemDataRole.UserRole
    try:
        page.load_form(_native_form(tmp_path, "multigeometry"))  # three outputs, each shown as it lands
        page.run_bar.startButton.click()
        _wait(page, qapp)
        paths = [page.results.item(row).data(role) for row in range(page.results.count())]
        assert set(page._payloads) == set(paths)  # all within the default budget: nothing to read back
        page.results.setCurrentRow(0)
        _wait(page, qapp)
        assert set(page._payloads) == set(paths) and page._reloading is None
        # a budget of one result keeps the shown one and the most recently shown other
        sizes = {path: page._payload_bytes[path] for path in paths}
        monkeypatch.setattr(module, "_RECENT_BYTES", max(sizes.values()))
        page.results.setCurrentRow(1)
        _wait(page, qapp)
        assert set(page._payloads) == {paths[1], paths[0]}  # paths[2] was shown least recently
        assert all(page._payload_bytes[path] == sizes[path] > 0 for path in page._payloads)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
