"""Stitch 2D on the Stitch page (STITCH-2D-20260930 B2; maintainer, 2026-09-29: "The 2d plot window should
have pretty much the and layout and buttons as int 2d without the raw 2d panel (the raw images still
accessible as in stitch 1d)"): Int 2D's layout without the raw pane, the map as the cake and I(q) below,
its slices and pins as Int 2D makes them from the same map, and the contributing frames as a Stitch 1D's.
"""
import numpy as np
import pytest

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


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_a_stitch_2d_shows_its_map_as_int_2d_without_the_raw_pane(qapp, tmp_path, backend):
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, backend, scans="1"))
        page.run_bar.modeCombo.setCurrentText("Stitch 2D")
        page.set_field("npt_rad_2d", "48")
        page.set_field("npt_azim_2d", "32")
        page.run_bar.startButton.click()
        _wait(page, qapp)
        target = page._owner.prepared.request.module.output.target
        assert target.endswith("_stitch2d.nexus")
        payload = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
        assert payload.intensity.shape == (48, 32)
        view, state = page.scientific, page._last_map_state
        assert view.raw.isHidden() and not view.cake.isHidden() and not view.image_splitter.isHidden()
        assert state.processing_mode == "Int 2D" and state.title == page.results.currentItem().text()
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
        assert page.frames.count() == 2
        row = payload.contributing_frames[0]
        frame = read_contributing_frame_view(target, row["identity"]["frame_index"],
                                             scan=row["key"].split("/")[0][5:], artifact_payload=payload)
        assert frame.thumbnail is not None
        document = page.settings()  # the mode is a setting, with the map's rows
        assert (document["mode"], document["fields"]["npt_rad_2d"]) == ("Stitch 2D", "48")
        # a Stitch 1D result still shows in Int 1D's layout
        page.run_bar.modeCombo.setCurrentText("Stitch 1D")
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page._current_payload().kind is AnalysisArtifactKind.STITCH_1D
        assert page._last_map_state is None and view.image_splitter.isHidden()
        page.apply_settings(document)
        assert page._mode == "Stitch 2D" and not page._map_card.isHidden()
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
