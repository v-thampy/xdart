"""Stitch's fit host captures its actual scientific selection and owns no fit widgets."""
from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from xrd_tools.io.analysis_artifact import read_analysis_artifact
from tests.xdart.pages.test_stitch_page import _Status, _wait, _published, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _command(page, kind, value=None):
    page._view_command(ShellCommand(kind, value))


def _finish(host, qapp):
    deadline = time.monotonic() + 15
    while host.computing and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
    qapp.processEvents()
    assert not host.computing


def _dispose(page, qapp):
    deadline = time.monotonic() + 15
    receipt = page.shutdown()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
        receipt = page.shutdown()
    assert receipt.status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def test_linear_current_background_and_real_shared_dialogs(qapp, tmp_path):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    form = _native_form(tmp_path, "multigeometry", scans="1")
    path, payload = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    try:
        assert not page.tools.page_tool("peak_fit").isEnabled()
        page.open_result(path)
        _wait(page, qapp)
        original = page._fit_current("peak").load()
        np.testing.assert_array_equal(original.values, payload.intensity)
        assert not original.values.flags.writeable
        scope = page._fit_scope()
        _command(page, ShellCommandKind.SET_LOG_SCALE, True)
        assert page._fit_scope() == scope
        np.testing.assert_array_equal(page._fit_current("peak").load().values, original.values)
        page.tools.page_tool("peak_fit").click()
        peak = page._fit_hosts["peak"]
        assert type(peak.dialog) is PeakFitDialog
        _finish(peak, qapp)
        np.testing.assert_array_equal(peak.dialog._y, payload.intensity)
        page.tools.page_tool("phase_analysis").click()
        phase = page._fit_hosts["phase"]
        assert type(phase.dialog) is PhaseFitDialog
        _finish(phase, qapp)
        radial = page._fit_current("phase").load()
        assert radial.axes[0].unit == "q_A^-1"
        assert radial.wavelength_angstrom == payload.wavelength_A
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "2theta")
        phase.reload()
        _finish(phase, qapp)
        assert phase.snapshot.axes[0].unit == "2th_deg"
        np.testing.assert_array_equal(phase.dialog._x, phase.snapshot.axes[0].values)
        from xrd_tools.notebooks import load_snapshot
        target = page._fit_current("phase")
        replay = load_snapshot(target.reference, bundle_root=tmp_path)
        assert replay.axes[0].unit == "2th_deg"
        np.testing.assert_array_equal(replay.values, phase.snapshot.values)
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "Q")
        page.scientific.background.click()
        subtracted = page._fit_current("peak").load()
        assert subtracted.key.background == "presentation" and page._fit_scope() != scope
        np.testing.assert_allclose(subtracted.values, 0., atol=1e-12)
        assert peak.snapshot is None  # changed selection releases the accepted detached input
        assert peak.dialog._x is None and peak.dialog._y is None
        assert not peak.dialog.fit_btn.isEnabled()
        page._diagnostic = "Coverage"
        page._draw()
        # PHASE-ANALYSIS-128: the Phase Analysis tool still opens (its list is the references); the fit's
        # reason moves to the window's Fit and Batch. The Peak Fitter keeps its own rule.
        assert not page.tools.page_tool("peak_fit").isEnabled()
        assert page.tools.page_tool("phase_analysis").isEnabled()
        assert not phase.dialog.fit_btn.isEnabled() and "FIT_INTENSITY_REQUIRED" in phase.dialog.fit_btn.toolTip()
        assert not phase.dialog.batch_btn.isEnabled()
        for kind in ("peak", "phase"):
            with pytest.raises(ValueError, match="FIT_INTENSITY_REQUIRED"):
                page._fit_current(kind)
    finally:
        _dispose(page, qapp)


def test_slice_and_matching_pin_are_the_actual_radial_trace(qapp, tmp_path):
    form = replace(_native_form(tmp_path, "xu_hist", scans="1"), mode="2d",
                   npt_rad_2d=48, npt_azim_2d=32)
    path, payload = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page.open_result(path)
        _wait(page, qapp)
        chi = payload.axis("chi")
        center, width = float(chi[len(chi) // 2]), 20.
        for kind, value in ((ShellCommandKind.SET_SLICE_CENTER, center),
                            (ShellCommandKind.SET_SLICE_WIDTH, width),
                            (ShellCommandKind.SET_SLICE_ENABLED, True)):
            _command(page, kind, value)
        expected = page._last_map_state.traces[0].intensity.copy()
        snap = page._fit_current("phase").load()
        np.testing.assert_array_equal(snap.values, expected)
        assert dict(snap.key.selection)["slice_enabled"] is True
        assert snap.sigma is None and snap.key.background == "none"
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        _command(page, ShellCommandKind.PIN_SLICE)
        assert page._last_map_state.traces == ()  # the identical pin absorbs the live curve
        np.testing.assert_array_equal(page._fit_current("peak").load().values, expected)
        _command(page, ShellCommandKind.SET_SLICE_CENTER, center + 30.)
        np.testing.assert_array_equal(page._fit_current("peak").load().values,
                                      page._last_map_state.traces[0].intensity)
        radial = page._last_map_state.heavy.cake_x.values
        _command(page, ShellCommandKind.SET_SHARE_AXIS, False)  # a shared map holds the plot axis (LF-011)
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "chi")
        _command(page, ShellCommandKind.SET_SLICE_CENTER, float(radial[len(radial) // 2]))
        _command(page, ShellCommandKind.SET_SLICE_WIDTH, float(abs(radial[-1] - radial[0]) / 4))
        assert page.tools.page_tool("phase_analysis").isEnabled()  # PHASE-ANALYSIS-128: the list still opens
        assert "PHASE_RADIAL_INTENSITY_REQUIRED" in page._fit_reason("phase")
        with pytest.raises(ValueError, match="PHASE_RADIAL_INTENSITY_REQUIRED"):
            page._fit_current("phase")
        np.testing.assert_array_equal(page._fit_current("peak").load().values,
                                      page._last_map_state.traces[0].intensity)
    finally:
        _dispose(page, qapp)


def test_gi_native_cut_is_peak_only(qapp, tmp_path):
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
    form = replace(_gi_form(tmp_path, "xu_hist", StitchGI(
        ("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)),
        mode="2d", npt_rad_2d=48, npt_azim_2d=32)
    path, payload = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page.open_result(path)
        _wait(page, qapp)
        _command(page, ShellCommandKind.SET_SHARE_AXIS, False)  # a shared map holds the plot axis (LF-011)
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "q_ip")
        cut = page._fit_current("peak").load()
        np.testing.assert_array_equal(cut.values, dict(payload.gi)["q_ip"].intensity)
        assert dict(cut.key.selection)["meaning"] == "q_ip"
        with pytest.raises(ValueError, match="PHASE_RADIAL_INTENSITY_REQUIRED"):
            page._fit_current("phase")
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        plain, radial = _published(replace(form, gi=None, output_path=plain_dir / "plain_stitch2d.nexus"))
        page.open_result(plain)
        _wait(page, qapp)
        _command(page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        for row in range(page.results.count()):
            page.results.item(row).setSelected(True)
        _command(page, ShellCommandKind.SET_PLOT_AXIS, "q_ip")
        # Int's mixed overlay keeps the radial result on Q. Its fit receipt names that actual axis.
        current = page._fit_current("peak").load()
        assert current.axes[0].name == "Q" and dict(current.key.selection)["meaning"] == "radial"
        assert dict(current.key.selection)["requested_coordinate"] == "q_ip"
        np.testing.assert_array_equal(current.values, radial.profile.intensity)
        assert page.tools.page_tool("phase_analysis").isEnabled()
    finally:
        _dispose(page, qapp)


def test_batch_selection_is_cheap_and_trend_reads_contributing_rows(qapp, tmp_path, monkeypatch):
    from xrd_tools.io import analysis_artifact
    form = _native_form(tmp_path, "xu_hist", scans="1")
    first, _ = _published(replace(form, mode="2d", npt_rad_2d=48, npt_azim_2d=32))
    # The batch is what the results browser lists, and since 063fc88f/8791ec8b the browser follows the opened
    # result's folder; Stitch names a result by its family, so the second one is its own family in the same folder.
    second, _ = _published(replace(form, mode="2d", output_family="second", npt_rad_2d=48, npt_azim_2d=32))
    assert Path(second).parent == Path(first).parent and Path(second).name != Path(first).name
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(form, mode="2d"))
        for path in (first, second):
            page.open_result(path)
            _wait(page, qapp)
        page.results.clearSelection()
        _wait(page, qapp)
        actual = analysis_artifact.read_analysis_artifact
        monkeypatch.setattr(analysis_artifact, "read_analysis_artifact",
                            lambda *a, **kw: pytest.fail("batch enumeration did IO on the GUI thread"))
        targets = page._fit_batch("peak")
        assert tuple(item.label for item in targets) == (Path(first).name, Path(second).name)
        monkeypatch.setattr(analysis_artifact, "read_analysis_artifact", actual)
        for path, target in zip((first, second), targets):
            np.testing.assert_array_equal(target.load().values, actual(path).profile.intensity)
            assert target.reference.result_fingerprint == actual(path).result_fingerprint
            from xrd_tools.notebooks import load_snapshot
            replayed = load_snapshot(target.reference, bundle_root=tmp_path)
            np.testing.assert_array_equal(replayed.values, target.load().values)
        trend = targets[0].trend("del")
        assert trend.available and trend.value == pytest.approx(14.5) and trend.count == 2
        assert not targets[0].trend("missing_column").available
        _command(page, ShellCommandKind.SET_SLICE_ENABLED, True)
        unavailable = page._fit_batch("peak")
        assert all(target.reference is None for target in unavailable)
        with pytest.raises(ValueError, match="FIT_BATCH_SELECTION_UNAVAILABLE"):
            unavailable[0].load()
        _command(page, ShellCommandKind.SET_SLICE_ENABLED, False)
        page._fit_batch("peak")[0].load()  # the native profile is available until BG is applied
        # Since 20e60fd2 Set BG takes the explicitly selected results; select the current one, as a user does.
        from pyqtgraph.Qt import QtCore
        current = page.browser.current_result()
        row = next(row for row in range(page.results.count())
                   if page.results.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == current)
        page.results.item(row).setSelected(True)
        _wait(page, qapp)
        page.scientific.background.click()
        assert page._background.phase == "ACTIVE"
        assert all(target.reference is None for target in page._fit_batch("peak"))
        # A result that is not displayed cannot capture the presentation background.
        other = next(str(path) for path in (first, second) if str(path) != str(current))
        with pytest.raises(ValueError, match="FIT_BATCH_SELECTION_UNAVAILABLE"):
            page._fit_target(other, "peak").load()
    finally:
        _dispose(page, qapp)


def test_reopen_invalidates_same_path_and_close_waits_for_real_worker(qapp, tmp_path, monkeypatch):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    path, original = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    entered, release = threading.Event(), threading.Event()
    try:
        page.open_result(path)
        _wait(page, qapp)
        stale = page._fit_current("peak")
        page._open_fit("peak")
        host = page._fit_hosts["peak"]
        _finish(host, qapp)
        old_scope = page._fit_scope()
        _, replacement = _published(replace(form, polarization_factor=.75))
        assert replacement.result_fingerprint != original.result_fingerprint
        page.open_result(path)
        _wait(page, qapp)
        assert page._fit_scope() != old_scope and host.snapshot is None
        assert host.dialog._y is None
        fresh = page._fit_current("peak").load()
        assert fresh.key.fingerprint == replacement.result_fingerprint
        np.testing.assert_array_equal(fresh.values, replacement.intensity)
        np.testing.assert_array_equal(stale.load().values, original.intensity)
        make = page._fit_target

        def delayed(path, kind, *, current=False):
            target = make(path, kind, current=current)

            def load():
                entered.set()
                assert release.wait(10)
                return target.load()

            return replace(target, load=load)

        monkeypatch.setattr(page, "_fit_target", delayed)
        host.reload()
        assert entered.wait(5)
        assert host.computing and page.active()
        assert "fit window" in page._settings_busy()
        assert page.shutdown().status is PageCleanup.PENDING
        release.set()
        _finish(host, qapp)
        assert page.shutdown().status is PageCleanup.CLEAN
        assert page._fit_projection is None and not page._payloads
    finally:
        release.set()
        _dispose(page, qapp)
