"""Stitch 2D on the Stitch page (STITCH-2D-20260930 B2; maintainer, 2026-09-29: "The 2d plot window should
have pretty much the and layout and buttons as int 2d without the raw 2d panel (the raw images still
accessible as in stitch 1d)"): Int 2D's layout without the raw pane, the map as the cake and I(q) below,
its slices and pins as Int 2D makes them from the same map, and the contributing frames as a Stitch 1D's.
"""
from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace
from pyqtgraph.Qt import QtCore, QtTest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.io.frame_view import read_contributing_frame_view
from xrd_tools.session.display_logic import nanmean_slice
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _command(page, kind, value=None):
    page._view_command(ShellCommand(kind, value))


@pytest.mark.parametrize("plot_mode", ["Single", "Overlay"])
def test_saved_stitch_2d_navigation_keeps_the_requested_curves(qapp, tmp_path, plot_mode, monkeypatch):
    """Real result clicks and footer visits agree with Int's selected scientific traces."""
    from xdart.gui.tabs.stitch import page as module
    monkeypatch.setattr(module, "_RECENT_BYTES", 0)
    page = StitchPage(empty_host_services(_Status()))
    page.resize(1400, 900)
    page.show()
    try:
        page.load_form(replace(_native_form(tmp_path, "multigeometry"), mode="2d",
                               npt_rad_2d=48, npt_azim_2d=32))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp, catalog=True)  # the paths below are read once the folder catalog has landed
        paths = page.browser.listed_results()  # directory rows are Browse navigation
        assert len(paths) == 3
        saved = {path: read_analysis_artifact(path, expected_kind=AnalysisArtifactKind.STITCH_2D)
                 for path in paths}
        keys = dict(page._result_keys)
        page.clear_plot()
        _command(page, ShellCommandKind.SET_PLOT_MODE, plot_mode)

        def click(row, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
            item = next(page.results.item(index) for index in range(page.results.count())
                        if page.results.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == paths[row])
            QtTest.QTest.mouseClick(page.results.viewport(), QtCore.Qt.MouseButton.LeftButton,
                                   modifier, page.results.visualItemRect(item).center())
            _wait(page, qapp)

        def shown(rows):
            expected = {paths[row] for row in rows}
            assert set(page.browser.selected_results()) == expected
            assert set(page._payloads) == expected | {page.browser.current_result()}
            traces = page._last_map_state.traces
            assert len(traces) == len(expected) == len(page.curve.listDataItems())
            for trace in traces:
                path = next(path for path, key in keys.items() if trace.frame is key)
                assert path in expected
                np.testing.assert_array_equal(trace.intensity, saved[path].profile.intensity)

        click(0)
        shown([0])
        click(1)
        shown([1] if plot_mode == "Single" else [0, 1])
        if plot_mode == "Single":
            click(2, QtCore.Qt.KeyboardModifier.ControlModifier)
            shown([1, 2])  # explicit selection is useful in Single too
            page.scientific.previous_frame.click()
            _wait(page, qapp)
            shown([1])
        else:
            page.scientific.next_frame.click()
            _wait(page, qapp)
            shown([0, 1, 2])
            click(1, QtCore.Qt.KeyboardModifier.ControlModifier)
            shown([0, 2])  # toggling membership keeps the current map separate
            QtTest.QTest.keyClick(page.results, QtCore.Qt.Key.Key_Up)
            _wait(page, qapp)
            shown([0, 2])  # revisiting a member never duplicates it
        assert all(page._result_keys[path] is key for path, key in keys.items())
        _command(page, ShellCommandKind.CLEAR_1D)
        assert page.browser.current_result() is None and page.browser.selected_results() == ()
        assert page._last_map_state.traces == () and page.curve.listDataItems() == []
        assert page._view.plot_mode == plot_mode
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("plot_mode", ["Single", "Overlay"])
def test_the_rows_a_run_lists_keep_their_order_when_the_catalog_lands(qapp, tmp_path, plot_mode, monkeypatch):
    """Nightly 39043642981's two reds: the folder catalog landed between reading the rows and a footer step. The rows
    listed while it is held keep their order after it lands; the footer and the no-selection fit batch walk them."""
    from xdart.gui.tabs.stitch import page as module
    from tests.xdart.pages.test_stitch_browser import _close, _held_catalog
    monkeypatch.setattr(module, "_RECENT_BYTES", 0)
    entered, release = _held_catalog(monkeypatch)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_native_form(tmp_path, "multigeometry"), mode="2d", npt_rad_2d=48, npt_azim_2d=32))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert entered.wait(10)
        held, keys = page.browser.listed_results(), dict(page._result_keys)
        current, selected = page.browser.current_result(), page.browser.selected_results()
        batch = [target.label for target in page._fit_batch("peak")]  # the run selected its result
        assert current and selected and batch == [Path(path).name for path in selected]
        release.set()
        _wait(page, qapp, catalog=True)
        assert {entry.artifact for entry in page._result_browser._owner.catalog} >= set(held)  # it landed
        assert len(held) == 3 and page.browser.listed_results() == held
        assert (page.browser.current_result(), page.browser.selected_results()) == (current, selected)
        assert all(page._result_keys[path] is key for path, key in keys.items())
        assert batch == [target.label for target in page._fit_batch("peak")]
        page.browser.clear_result_selection()
        _wait(page, qapp)
        assert [target.label for target in page._fit_batch("peak")] == [Path(path).name for path in held]
        _command(page, ShellCommandKind.SET_PLOT_MODE, plot_mode)
        page.browser.set_current_result(held[0], keep_selection=False)
        _wait(page, qapp)
        page.scientific.next_frame.click()
        _wait(page, qapp)
        expected = {held[1]} if plot_mode == "Single" else {held[0], held[1]}
        assert page.browser.current_result() == held[1] and set(page.browser.selected_results()) == expected
        assert [target.label for target in page._fit_batch("peak")] == [
            Path(path).name for path in page.browser.selected_results()]  # a selection batches in its own order
        traces = page._last_map_state.traces
        assert len(traces) == len(expected)
        for trace in traces:
            path = next(path for path, key in page._result_keys.items() if trace.frame is key)
            saved = read_analysis_artifact(path, expected_kind=AnalysisArtifactKind.STITCH_2D)
            assert path in expected
            np.testing.assert_array_equal(trace.intensity, saved.profile.intensity)
    finally:
        release.set()
        _close(page, qapp)


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_a_stitch_2d_shows_its_map_as_int_2d_without_the_raw_pane(qapp, tmp_path, backend):
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, backend, scans="1"))
        page.run_bar.modeCombo.setCurrentText("Stitch 2D")
        page.set_field("npt_rad_2d", "48")
        page.set_field("npt_azim_2d", "32")
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        target = page._owner.prepared.request.module.output.target
        assert target.endswith("_stitch2d.nexus")
        payload = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
        assert payload.intensity.shape == (48, 32)
        view, state = page.scientific, page._last_map_state
        assert view.raw.isHidden() and not view.cake.isHidden() and not view.image_splitter.isHidden()
        assert state.processing_mode == "Int 2D" and state.title == page.results.currentItem().text()
        assert state.share_axis and view.share_axis.isChecked()
        assert view.share_axis.isHidden()
        assert not view.plot_axis.isEnabled()  # shared, the plot follows the map
        # Off is an explicit viewing choice: later redraws must not reapply the default.
        view.options.click()
        assert view.plot_options_dialog.share_axis.isChecked()
        view.plot_options_dialog.share_axis.click()
        view.plot_options_dialog.accept_button.click()
        assert not page._view.share_axis and not page._last_map_state.share_axis
        assert view.plot_axis.isEnabled()
        page._draw()
        assert not view.share_axis.isChecked()
        view.options.click()
        assert not view.plot_options_dialog.share_axis.isChecked()
        view.plot_options_dialog.share_axis.click()
        view.plot_options_dialog.accept_button.click()
        assert page._view.share_axis and page._last_map_state.share_axis
        assert not view.plot_axis.isEnabled()
        cake = np.asarray(payload.intensity.T, dtype=float)  # (χ, q), as Int's frame holds a map
        np.testing.assert_array_equal(state.heavy.cake, cake)
        np.testing.assert_array_equal(state.traces[0].intensity, payload.profile.intensity)  # I(q) below
        assert state.traces[0].title == page.results.currentItem().text()
        # a slice: Int 2D's mean over the χ window, of the same map
        chi = payload.axis("chi")
        center, width = float(chi[len(chi) // 2]), 20.0
        for kind, value in ((ShellCommandKind.SET_SLICE_CENTER, center), (ShellCommandKind.SET_SLICE_WIDTH, width),
                            (ShellCommandKind.SET_SLICE_ENABLED, True)):
            _command(page, kind, value)
        window = (center - width <= chi) & (chi <= center + width)
        np.testing.assert_array_equal(page._last_map_state.traces[0].intensity, nanmean_slice(cake[window, :], 0))
        # a pin, in Overlay, as Int's: kept as the slice moves on
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        _command(page, ShellCommandKind.PIN_SLICE)
        _command(page, ShellCommandKind.SET_SLICE_CENTER, center + 30.0)
        assert len(page._view.slice_pins) == 1
        (pinned,) = page._last_map_state.pinned_traces
        np.testing.assert_array_equal(pinned.trace.intensity, nanmean_slice(cake[window, :], 0))
        # the contributing frames, as a Stitch 1D's
        assert page.frames.count() == 3
        row = payload.contributing_frames[0]
        frame = read_contributing_frame_view(target, row["identity"]["frame_index"],
                                             scan=row["key"].split("/")[0][5:], artifact_payload=payload)
        assert frame.thumbnail is not None
        document = page.settings()  # the mode is a setting, with the map's rows
        assert (document["mode"], document["fields"]["npt_rad_2d"]) == ("Stitch 2D", "48")
        # The layout is the mode's, as Int's follows its run-bar mode: a Stitch 1D result in Int 1D's
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Single")
        page.run_bar.modeCombo.setCurrentText("Stitch 1D")
        page.run_bar.startButton.click()
        _wait(page, qapp)
        one_d = page._current_payload()
        assert one_d.kind is AnalysisArtifactKind.STITCH_1D
        assert page._last_map_state is None and view.image_splitter.isHidden()
        # a Stitch 2D result there too, as its I(q) (Int 1D shows a 2-D record's I(q))
        role = QtCore.Qt.ItemDataRole.UserRole
        rows = {page.results.item(row).data(role): row for row in range(page.results.count())}
        page.results.setCurrentRow(rows[target])
        _wait(page, qapp)
        assert page._current_payload().kind is AnalysisArtifactKind.STITCH_2D
        assert page._last_map_state is None and view.image_splitter.isHidden()
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.profile.intensity)
        # and in Stitch 2D a Stitch 1D result shows in Int 2D's layout, its I(q) below an empty map
        page.apply_settings(document)
        assert page._mode == "Stitch 2D" and not page._map_card.isHidden()
        one_d_target = page._owner.prepared.request.module.output.target
        assert one_d_target not in page.browser.listed_results()  # Stitch 2D lists its own results by default
        page.browser.set_filename_filter("")  # All results
        rows = {page.results.item(row).data(role): row for row in range(page.results.count())}
        page.results.setCurrentRow(rows[one_d_target])
        _wait(page, qapp)
        assert page._current_payload().kind is AnalysisArtifactKind.STITCH_1D
        state = page._last_map_state
        assert state.processing_mode == "Int 2D" and not view.image_splitter.isHidden()
        assert state.traces == ()  # Int's rule: a slice is never the full I(q) of a result without a map
        _command(page, ShellCommandKind.SET_SLICE_ENABLED, False)
        (trace,) = page._last_map_state.traces
        np.testing.assert_array_equal(trace.intensity, one_d.intensity)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_the_layout_is_the_modes_as_int_2ds(qapp):
    """Maintainer, 2026-09-30: "when stitch 2d is selected, the plot display doesn't change ... to show two rows
    equally sized for the 2d stitch on top and 1d bottom, see how this is done in int 2d". The layout follows the
    mode, as Int's follows its run-bar mode (Int's ``_apply_mode_layout``), before any result."""
    page = StitchPage(empty_host_services(_Status()))
    page.resize(1400, 900)
    page.show()
    try:
        qapp.processEvents()
        view = page.scientific
        assert view.image_splitter.isHidden()  # Stitch 1D: Int 1D's layout
        page.run_bar.modeCombo.setCurrentText("Stitch 2D")
        qapp.processEvents()
        assert not view.image_splitter.isHidden() and page._last_map_state.processing_mode == "Int 2D"
        top, bottom = view.vertical_splitter.sizes()
        assert top > 0 and abs(top - bottom) <= 1  # Int 2D's two equal rows
        assert not view.viewer_intensity.isVisible()
        view.options.click()
        qapp.processEvents()
        assert view.viewer_intensity.isVisible() and view.plot_options_dialog.isAncestorOf(view.viewer_intensity)
        view.plot_options_dialog.close()
        page.run_bar.modeCombo.setCurrentText("Stitch 1D")
        qapp.processEvents()
        assert view.image_splitter.isHidden() and page._last_map_state is None
    finally:
        page.close()
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_the_2d_rows_are_int_2ds_and_reach_the_form_and_back(qapp, tmp_path):
    """Maintainer, 2026-09-30: "like int 1d/2d the Processing panel in Stitch should also have the capability of
    selecting a range for both axes in the 2d ... there is no choice to select the 2d integration axes". Int 2D's
    rows, from Int's composer: the 2-D Axis (Q-χ, 2θ-χ) with its Pts on it, and the map's radial and χ ranges,
    the radial one's label and defaults following the Axis; they reach the form (the map's own axis, I(q)
    keeping the 1-D one) and come back from it and from the settings."""
    from dataclasses import replace
    from xdart.gui.tabs.stitch.settings import apply_page_settings, page_settings
    page, other = StitchPage(empty_host_services(_Status())), StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_native_form(tmp_path, "multigeometry", scans="1"), mode="2d"))
        axis = page._rows["map_axis"]
        assert not page._map_card.isHidden() and page._map_card.isAncestorOf(axis) and axis.label.text() == "Axis"
        assert [axis.editor.itemText(index) for index in range(axis.editor.count())] == ["Q-χ", "2θ-χ"]
        assert all(axis.isAncestorOf(page._rows[key]) for key in ("npt_rad_2d", "npt_azim_2d"))  # Int's Pts
        assert page._rows["npt_1d"].maximumWidth() == page._rows["npt_rad_2d"].maximumWidth() == 58
        assert (page.map_range.label.text(), page.chi_range.label.text()) == ("Q (Å⁻¹)", "χ (°)")
        form = page.build_form()
        assert (form.map_unit, form.map_radial_range, form.chi_range) == ("q_A^-1", None, None)
        page.set_field("map_axis", "2θ-χ")
        assert page.map_range.label.text() == "2θ (°)" and page._fields["map_max"].value == 90.0
        page.set_field("map_auto", False)
        assert page.map_range._low.isEnabled()
        page.set_field("map_min", 12.0)
        page.set_field("map_max", 48.0)
        page.set_field("chi_auto", False)
        page.set_field("chi_min", -150.0)
        page.set_field("chi_max", -10.0)
        form = page.build_form()
        assert (form.map_unit, form.map_radial_range, form.chi_range) == ("2th_deg", (12.0, 48.0), (-150.0, -10.0))
        assert (form.unit, form.q_range) == ("q_A^-1", None)  # I(q) keeps the 1-D axis and range
        other.load_form(form)
        assert other._value("map_axis") == "2θ-χ" and other.build_form().fingerprint == form.fingerprint
        other.load_form(replace(form, map_unit="q_A^-1", map_radial_range=None, chi_range=None))
        apply_page_settings(other, page_settings(page))
        assert other.build_form().fingerprint == form.fingerprint
    finally:
        for each in (page, other):
            assert each.shutdown().status is PageCleanup.CLEAN
            each.deleteLater()
        qapp.processEvents()


def test_grazing_2d_axis_is_the_map_a_result_opens_on(qapp, tmp_path):
    """Int's Grazing 2-D axes (Qip-Qoop, Q-χ): a Grazing Stitch 2D writes both maps, in Q, and opens on the one
    chosen, as Grazing's 1-D Axis chooses the cut a result opens on."""
    from dataclasses import replace
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_gi_form(tmp_path, "multigeometry", StitchGI(
            ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)),
            mode="2d", npt_rad_2d=24, npt_azim_2d=16))
        axis = page._rows["map_axis"]
        assert [axis.editor.itemText(index) for index in range(axis.editor.count())] == ["Qip-Qoop", "Q-χ"]
        page.set_field("map_axis", "Qip-Qoop")
        assert page.build_form().map_unit == "q_A^-1"
        _wait(page, qapp)
        assert page.run_bar.startButton.isEnabled()
        page.run_bar.startButton.click()
        _wait(page, qapp)
        payload = page._current_payload()
        assert payload.gi_map is not None and page._view.image_axis == "qip_qoop"
        np.testing.assert_array_equal(page._last_map_state.heavy.cake, np.asarray(payload.gi_map.intensity.T, dtype=float))
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_a_gi_stitch_2d_shows_both_maps_as_ints_gi_view(qapp, tmp_path, backend):
    """Item 3: a GI Stitch 2D is a GI frame as Int's: the image axis offers the (chi, q) and the q_ip-q_oop
    maps, the cake is the one chosen, and the plot axis offers the cuts, a cut showing as stored."""
    from dataclasses import replace
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_gi_form(tmp_path, backend, StitchGI(
            ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)),
            mode="2d", npt_rad_2d=48, npt_azim_2d=32))
        assert page._mode == "Stitch 2D" and page._gi["enabled"]
        _wait(page, qapp)
        assert page.run_bar.startButton.isEnabled()
        page.run_bar.startButton.click()
        _wait(page, qapp)
        payload = page._current_payload()
        assert payload.kind is AnalysisArtifactKind.STITCH_2D and payload.gi_map is not None
        state = page._last_map_state
        assert state.measurement_mode == "GI" and tuple(state.gi_maps) == ("q_chi", "qip_qoop")
        np.testing.assert_array_equal(state.heavy.cake, np.asarray(payload.intensity.T, dtype=float))
        _command(page, ShellCommandKind.SET_IMAGE_AXIS, "qip_qoop")
        np.testing.assert_array_equal(page._last_map_state.heavy.cake,
                                      np.asarray(payload.gi_map.intensity.T, dtype=float))
        choices = [page.scientific.plot_axis.itemData(row) for row in range(page.scientific.plot_axis.count())]
        assert {"Q", "q_ip", "q_oop"} <= set(choices)
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "q_ip")
        np.testing.assert_array_equal(page._last_map_state.traces[0].intensity, dict(payload.gi)["q_ip"].intensity)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
