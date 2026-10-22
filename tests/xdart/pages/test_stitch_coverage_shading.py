"""AZ-COVER: shared display support shading, without changing scientific arrays."""
from dataclasses import replace

import numpy as np
import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog, plot_option_update
from xdart.gui.tabs.scattering.shell_values import AxisProjection, HeavyProjection, ScientificPlotOptions
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane
from xdart.gui.tabs.stitch.page import StitchPage
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.tools.test_stitch_native_values import _native_form


def _axes():
    return AxisProjection(np.array([1., 2., 3.]), "Q", "Å⁻¹"), AxisProjection(np.array([20., -20.]), "chi", "deg")


@pytest.mark.parametrize("coverage", [np.zeros((3, 2)), np.full((2, 3), -1.),
                                      np.full((2, 3), np.nan), np.zeros((2, 3), dtype=bool)])
def test_coverage_projection_refuses_invalid_support(coverage):
    frame = make_shell_projection(frame_count=1, heavy_indices=(0,)).scientific.heavy.frame
    x, y = _axes()
    with pytest.raises((TypeError, ValueError), match="coverage"):
        HeavyProjection(frame, cake=np.ones((2, 3)), cake_x=x, cake_y=y, cake_coverage=coverage)


def test_shared_pane_shades_zero_support_in_image_orientation(qapp):
    pane = ScientificImagePane(lock_aspect=False)
    data = np.array([[7., 0., 3.], [2., 5., 8.]])
    coverage = np.array([[0., .25, 1.], [1., 0., 0.]])
    before = data.copy(), coverage.copy()
    x, y = _axes()
    try:
        pane.render(data, x_axis=x, y_axis=y)
        pane.render_coverage(coverage)
        overlay = pane.coverage_overlay
        assert overlay.isVisible()
        np.testing.assert_array_equal(overlay.image[..., 3] > 0, coverage.T == 0)
        np.testing.assert_array_equal(overlay.transform().mapRect(overlay.boundingRect()),
                                      pane.image.transform().mapRect(pane.image.boundingRect()))
        assert [reference() for reference in pane.color_scale.img_list] == [pane.image]
        np.testing.assert_array_equal(data, before[0])
        np.testing.assert_array_equal(coverage, before[1])
        overlay.render()  # force the real pyqtgraph QImage path without native painting
        assert overlay.qimage is not None
        pane.render_coverage(None)
        assert overlay.image is None and not overlay.isVisible()
        assert overlay.qimage is None and overlay._displayBuffer is None
        pane.render_coverage(coverage)
        pane.clear_image_buffers()
        assert overlay.image is None and not overlay.isVisible()
    finally:
        pane.deleteLater()
        qapp.processEvents()


def test_shared_coverage_option_defaults_off_and_cancel_restores(qapp):
    options = ScientificPlotOptions()
    assert not options.show_coverage_holes
    assert plot_option_update(options, ("other", "coverage_holes"), True).show_coverage_holes
    assert plot_option_update(options, ("other", "coverage_holes"), 1) is None
    dialog = PlotOptionsDialog()
    try:
        assert not dialog.show_coverage_holes.isEnabled()
        dialog.set_coverage_available(True)
        dialog.show_coverage_holes.setChecked(True)
        assert dialog.edited_options().show_coverage_holes
        dialog.reject()
        assert not dialog.show_coverage_holes.isChecked()
        assert "pixel splitting" in dialog.show_coverage_holes.toolTip()
        dialog.set_coverage_available(False)
        assert "Stitch Q–χ intensity" in dialog.show_coverage_holes.toolTip()
    finally:
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_saved_stitch_coverage_follows_options_diagnostic_and_mode(qapp, tmp_path, backend):
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_native_form(tmp_path, backend, scans="1"), mode="2d",
                               npt_rad_2d=48, npt_azim_2d=32))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        target = page._owner.prepared.request.module.output.target
        saved = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
        intensity, coverage = saved.intensity.copy(), saved.coverage.copy()
        overlay = page.scientific.cake.coverage_overlay
        dialog = page.scientific.plot_options_dialog
        assert not overlay.isVisible() and dialog.show_coverage_holes.isEnabled()
        dialog.show_coverage_holes.setChecked(True)
        dialog.accept_button.click()
        assert page._view.plot_options.show_coverage_holes
        np.testing.assert_array_equal(page._last_map_state.heavy.cake_coverage, saved.coverage.T)
        np.testing.assert_array_equal(overlay.image[..., 3] > 0, saved.coverage == 0)
        # Cancel is a real dialog edit: it never changes the accepted option.
        dialog.show_coverage_holes.setChecked(False)
        dialog.cancel_button.click()
        assert dialog.show_coverage_holes.isChecked() and overlay.isVisible()
        page._diagnostic_combo.setCurrentText("Coverage")
        dialog.accept_button.click()
        assert overlay.image is None and not dialog.show_coverage_holes.isEnabled()
        page._diagnostic_combo.setCurrentText("Intensity")
        dialog.accept_button.click()
        assert overlay.isVisible() and dialog.show_coverage_holes.isEnabled()
        dialog.show_coverage_holes.setChecked(False)
        dialog.accept_button.click()
        assert overlay.image is None and not overlay.isVisible()
        dialog.show_coverage_holes.setChecked(True)
        dialog.accept_button.click()
        page.run_bar.modeCombo.setCurrentText("Stitch 1D")
        assert overlay.image is None and not dialog.show_coverage_holes.isEnabled()
        page.run_bar.modeCombo.setCurrentText("Stitch 2D")
        assert overlay.isVisible()
        page.clear_plot()
        assert overlay.image is None and not dialog.show_coverage_holes.isEnabled()
        page.open_result(target)
        _wait(page, qapp)
        assert overlay.isVisible()
        reopened = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
        np.testing.assert_array_equal(reopened.intensity, intensity)
        np.testing.assert_array_equal(reopened.coverage, coverage)
        overlay.render()
        assert overlay.qimage is not None
        assert page.shutdown().status is PageCleanup.CLEAN
        assert overlay.image is None and not overlay.isVisible()
        assert overlay.qimage is None and overlay._displayBuffer is None
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_gi_coverage_stays_on_primary_map_only(qapp, tmp_path, backend):
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_gi_form(tmp_path, backend, StitchGI(
            ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)),
            mode="2d", npt_rad_2d=24, npt_azim_2d=16))
        _wait(page, qapp)  # actual current readiness includes the detector observer
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page._current_payload().gi_map is not None
        dialog = page.scientific.plot_options_dialog
        overlay = page.scientific.cake.coverage_overlay
        dialog.show_coverage_holes.setChecked(True)
        dialog.accept_button.click()
        assert overlay.isVisible() and dialog.show_coverage_holes.isEnabled()
        page._view_command(ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "qip_qoop"))
        assert overlay.image is None and not dialog.show_coverage_holes.isEnabled()
        page._view_command(ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "q_chi"))
        assert overlay.isVisible() and dialog.show_coverage_holes.isEnabled()
    finally:
        _wait(page, qapp)
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
