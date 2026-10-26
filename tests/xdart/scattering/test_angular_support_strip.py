"""COVERAGE-SUPPORT-20261013 frozen GUI oracle (Claude, October 13, after Codex's ratification).

Rows D1-D8 of the kickoff (handoffs/coverage_support_2026-10-13.md): real Runs through the production executor and
page, the Run's own terminal handoff into Browse, real reopen in a fresh page, the real 1D Viewer and the real Stitch
page. Pages mount the shared ScientificView strip, Plot Options control and hover; no row builds a widget. New APIs
are imported inside each row, so a missing API fails only its own row; the parent results are recorded in the
kickoff's freeze record. Expected values come from tests/core/test_angular_support.py's independent models.
"""
import math
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from tests.core.test_angular_support import TOL, _deg, _expected, _planar, _rect
from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_browse_reference_rings_run import (
    SAVED, _browsing, _page, _pump, _reopen, _run, _series, needs_data)
from tests.xdart.scattering.test_predicted_ring_overlays import qapp, select  # noqa: F401
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

PERPENDICULAR = ('poni_version: 2.1\nDetector: Pilatus100k\nDetector_config: {"orientation": 3}\n'
                 'Distance: 0.05\nPoni1: 0.016813\nPoni2: 0.041907\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\n'
                 'Wavelength: 1.0e-10\n')
BAND = (300, 306)


# ---------------------------------------------------------------------------------------------------- helpers

def _enable(qapp, view, on=True):
    """Toggle the shared option through the real dialog and its Okay button."""
    view.options.click()
    dialog = view.plot_options_dialog
    dialog.show_angular_support.setChecked(on)
    dialog.accept_button.click()
    qapp.processEvents()


def _curves(view):
    """The strip's drawn curves by trace title, each with the profile object it was drawn from."""
    plot = view.angular_support_plot
    return {item.name(): (np.column_stack(item.getData()), item.angular_support)
            for item in plot.getPlotItem().listDataItems() if item.getData()[0] is not None}


def _wait(qapp, predicate, seconds=30.):
    assert _pump(qapp, predicate, seconds), 'the strip never settled'


def _command(qapp, page, kind, value=None):
    from xdart.gui.tabs.scattering.shell_values import ShellCommand
    page._handle_shell_command(ShellCommand(kind, value))
    page._refresh_shell()
    qapp.processEvents()


def _hover(view):
    """Hover lines from every strip item that provides them, at the strip's horizontal centre."""
    plot = view.angular_support_plot
    box = plot.getPlotItem().getViewBox()
    (x0, x1), _y = box.viewRange()
    point = QtCore.QPointF(.5 * (x0 + x1), 50.)
    lines = []
    for item in plot.getPlotItem().items:
        entries = getattr(item, 'hover_entries', None)
        if entries is not None:
            lines.extend(entries(point, box.mapViewToScene(point)))
    return lines


def _masked_series(root, *, mode='Int 1D', unit='2th_deg', energies=(16, 16), output_mode='Overwrite', one=None,
                   band=BAND):
    """A Run of a perpendicular Pilatus 100K with one masked column band (analytic expectation). Coverage science
    asks for recording explicitly, Off being the default since INT-PERF-REGRESSION-20261014; a row about the default
    itself removes the option."""
    root.mkdir(parents=True, exist_ok=True)
    intent = _series(root, energies, mode=mode, unit=unit, one={'npt': 32, 'method': 'csr', **(one or {})},
                     poni_text=PERPENDICULAR, output_mode=output_mode)
    mask = np.zeros((195, 487), np.uint8)
    mask[:, band[0]:band[1]] = 1
    np.save(root / 'mask.npy', mask)
    intent.mask_file = str(root / 'mask.npy')
    intent.run_options['record_angular_support'] = True
    return intent


def _analytic(profile, band=BAND):
    """The independent bin-mean support of ``profile``'s own bins for the perpendicular masked-band Run."""
    from pyFAI.detectors import detector_factory
    ai = _planar(dist=0.05, poni=(0.016813, 0.041907), detector=detector_factory('Pilatus100k'),
                 wavelength=profile.provenance['wavelength_m'])
    rect = _rect(ai, 0, 195, 0, 487)
    return _expected(profile.edges, profile.radial_unit, ai, rect, [_rect(ai, 0, 195, band[0], band[1])])


def _fresh(qapp, root, **series):
    """Run to terminal, browse the output, return its path."""
    with _page(qapp, _masked_series(root, **series)) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
    return written(root / 'run.nexus', series.get('mode', 'Int 1D'))


# ---------------------------------------------------------------------------------------------------- D1

@pytest.mark.parametrize('mode,unit', (('Int 1D', '2th_deg'), ('Int 1D', 'q_A^-1'), ('Int 2D', '2th_deg')))
def test_d1_option_and_real_int_run(qapp, tmp_path, mode, unit):
    with _page(qapp, _masked_series(tmp_path, mode=mode, unit=unit)) as (page, view, observed):
        dialog = view.plot_options_dialog
        assert not dialog.show_angular_support.isChecked(), 'the strip is opt-in'
        assert dialog.show_angular_support.toolTip(), 'the option explains itself'
        assert not view.angular_support_plot.isVisible()
        from xrd_tools.core.angular_support import SupportOrigin
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        # Browse's Int 2D curve is a saved cake cut, its own asynchronous slice: the strip can paint first
        # (hosted run 40289672559: strip {'escan_2'}, traces empty).
        _wait(qapp, lambda: bool(_curves(view)) and bool(page._last_scientific_projection.traces))
        assert view.angular_support_plot.isVisible()
        curves = _curves(view)
        traces = page._last_scientific_projection.traces
        assert set(curves) == {trace.title for trace in traces}, 'one strip curve per native trace'
        for trace in traces:
            data, profile = curves[trace.title]
            assert profile.origin is SupportOrigin.SAVED and profile.available
            np.testing.assert_allclose(data[:, 0], trace.axis.values, rtol=1e-6)
            np.testing.assert_allclose(data[:, 1], 100 * profile.supported_deg / profile.window_deg, atol=1e-9)
            expected = _analytic(profile)
            assert np.all(np.abs(profile.supported_deg - expected) <= TOL * 360), trace.title
        assert view.angular_support_plot.getPlotItem().getViewBox().viewRange()[1] == [0.0, 100.0]
        _enable(qapp, view, on=False)
        _wait(qapp, lambda: not view.angular_support_plot.isVisible())


def test_d1b_a_gi_total_q_run_strips_chi_gi(qapp, tmp_path):
    from xrd_tools.session.run_configuration import GIIntent
    intent = _masked_series(tmp_path, unit='q_A^-1')
    intent.gi = GIIntent(enabled=True, incidence_motor='Manual', th_val=1.0, mode_1d='q_total')
    with _page(qapp, intent) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        _wait(qapp, lambda: bool(_curves(view)))
        (data, profile), = _curves(view).values()
        assert profile.available and profile.convention == 'gi_chi' and profile.window_deg == 360.


# ---------------------------------------------------------------------------------------------------- D2

def test_d2_strip_alignment_and_transforms(qapp, tmp_path):
    with _page(qapp, _masked_series(tmp_path)) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        _wait(qapp, lambda: bool(_curves(view)))
        baseline = {title: data.copy() for title, (data, _profile) in _curves(view).items()}
        strip_box = view.angular_support_plot.getPlotItem().getViewBox()
        curve_box = view.curve.getPlotItem().getViewBox()
        for low, high, width in ((10., 20., 900), (12.5, 13.5, 600), (5., 30., 1200)):
            page.resize(width, 900)
            curve_box.setXRange(low, high, padding=0)
            qapp.processEvents()
            assert strip_box.viewRange()[0] == pytest.approx(curve_box.viewRange()[0], abs=1e-9)
            for x in (low, .5 * (low + high), high):
                strip_px = strip_box.mapViewToScene(QtCore.QPointF(x, 0)).x()
                curve_px = curve_box.mapViewToScene(QtCore.QPointF(x, 0)).x()
                assert abs(strip_px - curve_px) <= 1.0, (x, strip_px, curve_px)
            assert strip_box.viewRange()[1] == [0.0, 100.0], 'the percentage axis never rescales'
        # intensity transforms and overlays never transform support
        dialog = view.plot_options_dialog
        for scale in ('Log', 'Sqrt', 'Linear'):
            view.options.click()
            dialog.intensity_scale.setCurrentText(scale)
            dialog.accept_button.click()
            qapp.processEvents()
            for title, (data, _profile) in _curves(view).items():
                np.testing.assert_array_equal(data, baseline[title])


# ---------------------------------------------------------------------------------------------------- D3

def test_d3_availability_follows_selected_trace(qapp, tmp_path):
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    from xrd_tools.core.angular_support import REASON_TEXT, SupportReason
    with _page(qapp, _masked_series(tmp_path, mode='Int 2D', energies=(16, 16, 16))) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        _wait(qapp, lambda: bool(_curves(view)))
        # a display unit conversion keeps each bin's own result on the converted axis. Test-only oracle repair: an
        # Int 2D Browse opens on Q whatever the run's unit (the plot-axis preference), so the row first shows the
        # run's own 2θ axis, then converts.
        _command(qapp, page, ShellCommandKind.SET_PLOT_AXIS, '2theta')
        _wait(qapp, lambda: page._last_scientific_projection.plot_axis == '2theta' and bool(_curves(view)))
        (native, profile), = _curves(view).values()
        _command(qapp, page, ShellCommandKind.SET_PLOT_AXIS, 'Q')
        _wait(qapp, lambda: bool(_curves(view)) and not np.allclose(
            next(iter(_curves(view).values()))[0][:, 0], native[:, 0]))
        (converted, same), = _curves(view).values()
        np.testing.assert_array_equal(converted[:, 1], native[:, 1])
        assert same is profile
        _command(qapp, page, ShellCommandKind.SET_PLOT_AXIS, '2theta')
        # aggregates and a cake-derived slice have reasons; the option intent stays checked
        keys = page._context_controller.frame_keys
        assert page._context_controller.select_navigation(keys[0], tuple(keys))
        for kind, value, reason in ((ShellCommandKind.SET_PLOT_MODE, 'Average', SupportReason.AGGREGATE_TRACE),
                                    (ShellCommandKind.SET_PLOT_MODE, 'Sum', SupportReason.AGGREGATE_TRACE),
                                    (ShellCommandKind.SET_PLOT_MODE, 'Waterfall', SupportReason.AGGREGATE_TRACE)):
            _command(qapp, page, kind, value)
            _wait(qapp, lambda: REASON_TEXT[reason] in view.angular_support_message())
            assert not _curves(view)
            assert view.plot_options_dialog.show_angular_support.isChecked()
        _command(qapp, page, ShellCommandKind.SET_PLOT_MODE, 'Single')
        assert page._context_controller.select_navigation(keys[0], (keys[0],))
        # Test-only oracle repair: this Run's cake has four χ bins (±45°, ±135°), so the cut is centred on one (a cut
        # at the default χ = 0 ± 10° holds no bin and makes no trace)
        _command(qapp, page, ShellCommandKind.SET_SLICE_CENTER, 45.0)
        _command(qapp, page, ShellCommandKind.SET_SLICE_ENABLED, True)
        _wait(qapp, lambda: REASON_TEXT[SupportReason.NOT_RADIAL_PROFILE] in view.angular_support_message())
        assert view.plot_options_dialog.show_angular_support.isChecked()
        _command(qapp, page, ShellCommandKind.SET_SLICE_ENABLED, False)
        _wait(qapp, lambda: bool(_curves(view)))


@needs_data
def test_d3b_saved_average_and_reintegrate_outputs_name_their_reasons(qapp, tmp_path):
    from xrd_tools.core.angular_support import REASON_TEXT, SupportReason
    from tests.xdart.scattering.test_browse_reference_rings_run import _directory
    with _page(qapp, _directory(tmp_path, live=False)) as (page, view, _observed):
        _enable(qapp, view)
        for name, reason in (('eiger_S069Ta_redo_eta2p0_1_scan001_master_average.nexus', SupportReason.AVERAGE_RESULT),
                             ('eiger_S069Ta_redo_eta2p0_1_scan001_reintegrate1d.nexus',
                              SupportReason.LEGACY_REINTEGRATE)):
            copy = tmp_path / name
            shutil.copy2(SAVED / name, copy)
            _wait(qapp, lambda: not page._context_controller.browse_preview_polling_needed)
            _reopen(qapp, page, copy)
            _wait(qapp, lambda: REASON_TEXT[reason] in view.angular_support_message())
            assert not _curves(view)


# ---------------------------------------------------------------------------------------------------- D4

def test_d4_saved_browse_and_successor(qapp, tmp_path):
    from xrd_tools.core.angular_support import SupportOrigin
    output = _fresh(qapp, tmp_path / 'first')
    # fresh page, own scratch copy, original raw/calibration/mask hidden
    copy = tmp_path / 'copy.nexus'
    shutil.copy2(output, copy)
    hidden = tmp_path / 'hidden'
    shutil.move(str(tmp_path / 'first'), str(hidden))
    other = _masked_series(tmp_path / 'other', band=(20, 120), one={'npt': 64})
    with _page(qapp, other) as (page, view, _observed):
        _enable(qapp, view)
        _reopen(qapp, page, copy)
        _wait(qapp, lambda: bool(_curves(view)))
        (data, profile), = _curves(view).values()
        assert profile.origin is SupportOrigin.SAVED and len(profile.supported_deg) == 32
        expected = _analytic(profile)
        assert np.all(np.abs(profile.supported_deg - expected) <= TOL * 360), 'current controls/mask cannot reach it'
    shutil.move(str(hidden), str(tmp_path / 'first'))

    # Replace with another mask: the new output's support, never the cached old one
    with _page(qapp, _masked_series(tmp_path / 'first', band=(100, 200))) as (page, view, observed):
        _enable(qapp, view)
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _wait(qapp, lambda: bool(_curves(view)))
        (data, profile), = _curves(view).values()
        np.testing.assert_allclose(profile.supported_deg, _analytic(profile, (100, 200)),
                                   atol=TOL * 360)

    # Append adds its own epoch's support for its own rows
    with _page(qapp, _masked_series(tmp_path / 'first', energies=(16, 16, 16), output_mode='Append',
                                    band=(100, 200))) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
    from xrd_tools.io.frame_view import FrameViewReader
    with FrameViewReader(written(tmp_path / 'first' / 'run.nexus', 'Int 1D')) as reader:
        labels = reader.labels()
        assert len(labels) >= 3
        assert all(reader.read(label).angular_support.available for label in labels)

    # a Reintegrate successor recomputes support for its own science, never copying its predecessor's. Test-only
    # oracle repair: Reintegrate re-integrates the output's persisted mask (its preparation carries only the current
    # integration settings, ScatteringWorkspace._reintegrate_preparation; the successor loads the record's mask,
    # reintegrate_successor.py _load_persisted_mask, as at cf49f3cb), so its own 48 bins, not the predecessor's 32,
    # tell a recomputation from a copy.
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    artifact = written(tmp_path / 'first' / 'run.nexus', 'Int 1D')
    intent = _masked_series(tmp_path / 'first', band=(100, 200), one={'npt': 48})
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry='entry', dimension='1d', source_root=str(tmp_path / 'first'),
        expected_target_snapshot=capture_target_snapshot(artifact),
        expected_labels=tuple(int(v) for v in open_scan(artifact).frame_indices),
        preparation=ScatteringWorkspace._reintegrate_preparation(intent, '1d'))
    result = run_reintegrate_successor(plan)
    assert result.disposition == 'COMMITTED', result
    with FrameViewReader(Path(result.output_artifact)) as reader:
        successor = reader.read(reader.labels()[0]).angular_support
    assert successor.available and len(successor.supported_deg) == 48
    np.testing.assert_allclose(successor.supported_deg, _analytic(successor, (100, 200)),
                               atol=TOL * 360)


# ---------------------------------------------------------------------------------------------------- D5

def _legacy_copy(source, target):
    """A copy of a fresh output with its saved support removed: the older-output form."""
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    shutil.copy2(source, target)
    with h5py.File(target, 'r+', locking=False) as handle:
        group = handle['entry/integrated_1d']
        for name in ANGULAR_SUPPORT_DATASETS:
            del group[name]
    return target


def test_d5_legacy_reconstruction_coalesces_and_cleans_up(qapp, tmp_path):
    from xrd_tools.core.angular_support import REASON_TEXT, SupportOrigin, SupportReason
    output = _fresh(qapp, tmp_path / 'run', energies=(16, 16, 16))
    legacy = _legacy_copy(output, tmp_path / 'legacy.nexus')
    with _page(qapp, _masked_series(tmp_path / 'other')) as (page, view, _observed):
        _enable(qapp, view)
        _reopen(qapp, page, legacy)
        owner = view.angular_support_owner
        keys = page._context_controller.frame_keys
        # deterministic supersession: ask for every frame quickly; only the last selection is drawn
        for key in keys:
            select(qapp, page, key)
        _wait(qapp, lambda: REASON_TEXT[SupportReason.PENDING] in view.angular_support_message()
              or bool(_curves(view)))
        assert REASON_TEXT[SupportReason.PENDING] in view.angular_support_message() or _curves(view)
        _wait(qapp, lambda: bool(_curves(view)) and not owner.computing)
        (data, profile), = _curves(view).values()
        assert profile.origin is SupportOrigin.RECONSTRUCTED and profile.available
        expected = _analytic(profile)
        assert np.all(np.abs(profile.supported_deg - expected) <= TOL * 360)
        assert owner.retained_bytes() <= 64 << 20 and owner.retained_keys() <= 8
        # unchanged activation reuses the completed key; off, clear and close retire to CLEAN
        computed = owner.completed_requests()
        select(qapp, page, keys[-1])
        assert owner.completed_requests() == computed
        _enable(qapp, view, on=False)
        assert owner.shutdown().status.name == 'CLEAN'
        assert not _curves(view)


# ------------------------------------------------------------------- W7L (PRE-RELEASE-CORRECTNESS-20261015)

# The older-form copy a real ae6f0453 Run wrote (retained leaf); its row exists only when it is named.
PARENT_SEAM_RECORD = os.environ.get('XDART_CHI_SEAM_PARENT_RECORD')


@pytest.mark.parametrize('record', ('current',) + (('parent',) if PARENT_SEAM_RECORD else ()))
def test_w7l_older_seam_window_output_refuses_reconstruction(qapp, tmp_path, record):
    """An older q output with a shifted full-turn chi window cannot show which directions its
    reducer admitted; matching saved q bins never make that full support."""
    from xrd_tools.analysis.angular_support import SavedSupportRequest, reconstruct_saved_support
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.output_transaction import capture_target_snapshot
    if record == 'parent':
        legacy = Path(shutil.copy2(PARENT_SEAM_RECORD, tmp_path / 'legacy.nexus'))
    else:
        output = _fresh(qapp, tmp_path / 'run', unit='q_A^-1', one={
            'radial_range': [0.5, 3.0], 'azimuth_range': [-180.0, 180.0], 'chi_offset': 30.0})
        legacy = _legacy_copy(output, tmp_path / 'legacy.nexus')
    with FrameViewReader(legacy) as reader:
        labels = tuple(reader.labels())
    request = SavedSupportRequest(str(legacy), 'entry', capture_target_snapshot(legacy), labels, labels)
    profiles = reconstruct_saved_support(request)
    assert labels and sorted(profiles) == sorted(labels)
    for profile in profiles.values():
        assert profile.origin is SupportOrigin.RECONSTRUCTED and not profile.available
        assert profile.reason is SupportReason.LEGACY_INSUFFICIENT
        assert profile.detail == 'historical chi-window admission cannot be established'


# ---------------------------------------------------------------------------------------------------- D6

def test_d6_shared_consumers_and_tooltips(qapp, tmp_path):
    from xdart.gui.tabs.scattering.scientific_plot_options import ANGULAR_SUPPORT_TOOLTIP, COVERAGE_TOOLTIP
    from xrd_tools.core.angular_support import REASON_TEXT
    assert 'angular' in ANGULAR_SUPPORT_TOOLTIP.lower() and 'pixel' in ANGULAR_SUPPORT_TOOLTIP.lower()
    assert ANGULAR_SUPPORT_TOOLTIP != COVERAGE_TOOLTIP
    assert all(text for text in REASON_TEXT.values())
    with _page(qapp, _masked_series(tmp_path)) as (page, view, observed):
        dialog = view.plot_options_dialog
        dialog.set_angular_support_available(False, 'No native radial profile is shown.')
        assert dialog.show_angular_support.toolTip() and 'No native radial profile' in \
            dialog.show_angular_support.toolTip()
        dialog.set_angular_support_available(True)
        assert dialog.show_angular_support.toolTip() == ANGULAR_SUPPORT_TOOLTIP
        assert view.angular_support_plot.toolTip(), 'the strip says what it measures'
        assert 'Shade uncovered bins' in [box.text() for box in dialog.findChildren(type(dialog.show_angular_support))]
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        _wait(qapp, lambda: bool(_curves(view)))
        hover = _hover(view)
        assert hover and all('over this bin' in line and '°' in line and '%' in line for line in hover)
    # the shared owners are mounted by the 1D Viewer and Stitch pages only, never re-created
    from tests.xdart.pages.test_gui_reuse_fence import test_page_modules_build_no_display_or_editor_widgets_of_their_own
    test_page_modules_build_no_display_or_editor_widgets_of_their_own()


# ---------------------------------------------------------------------------------------------------- D7

def test_d7_compare_saved_profiles(qapp, tmp_path):
    """Each profile keeps its own grid, window and colour; a source without a record is named, hiding nothing."""
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    from xrd_tools.core.angular_support import REASON_TEXT, SupportReason
    from tests.xdart.scattering.test_viewer_1d_selection import _ready
    # the 1D Viewer's text sources have no record: individually unavailable, the option intent kept
    paths = tuple(str(tmp_path / f'curve_{index}.xye') for index in range(3))
    for index, path in enumerate(paths):
        np.savetxt(path, np.column_stack((np.arange(3.), np.arange(3.) + 10 * (index + 1), np.ones(3))))
    viewer_intent = _masked_series(tmp_path / 'viewer')
    viewer_intent.processing_mode = '1D Viewer'
    with _page(qapp, viewer_intent) as (page, view, _observed):
        page._set_browser_directory(str(tmp_path), explicit=True)
        page._open_viewer_1d_paths(paths, current_path=paths[-1])
        _ready(qapp, page)
        _enable(qapp, view)
        _wait(qapp, lambda: REASON_TEXT[SupportReason.NO_RECORD] in view.angular_support_message())
        assert not _curves(view)
    # Int Browse: an energy scan's frames compared in Overlay, one curve per frame with its own support and colour
    with _page(qapp, _masked_series(tmp_path / 'scan', unit='q_A^-1', energies=(14, 16, 18))) as (page, view,
                                                                                                   observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _enable(qapp, view)
        _command(qapp, page, ShellCommandKind.SET_PLOT_MODE, 'Overlay')
        keys = page._context_controller.frame_keys
        assert page._context_controller.select_navigation(keys[0], tuple(keys))
        page._refresh_shell()
        _wait(qapp, lambda: len(_curves(view)) == 3)
        curves = _curves(view)
        profiles = [profile for _data, profile in curves.values()]
        assert len({profile.supported_deg.tobytes() for profile in profiles}) == 3, 'no borrowed support'
        colours = {item.name(): item.opts['pen'].color().name()
                   for item in view.angular_support_plot.getPlotItem().listDataItems()}
        traces = {item.name(): item.opts['pen'].color().name() for item in view.curve.getPlotItem().listDataItems()}
        assert colours and all(colours[title] == traces[title] for title in colours), 'each curve keeps its colour'
        assert page._context_controller.select_navigation(keys[0], tuple(keys[:2]))
        page._refresh_shell()
        _wait(qapp, lambda: len(_curves(view)) == 2)


# ---------------------------------------------------------------------------------------------------- D8

def test_d8_stitch_run_browse_and_exclusions(qapp, tmp_path):
    from dataclasses import replace
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.core.test_stitch_operation import _native_form
    from tests.xdart.pages.test_stitch_page import _Status, _wait as _stitch_wait
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(replace(_native_form(tmp_path, 'multigeometry', scans='1'), mode='1d'))
        _stitch_wait(page, qapp)
        # coverage science asks for recording through the real control (Off by default since INT-PERF-REGRESSION-20261014)
        from pyqtgraph.Qt import QtWidgets
        pill, = [button for button in page.findChildren(QtWidgets.QAbstractButton)
                 if button.isCheckable() and button.accessibleName() == 'Record angular coverage']
        if not pill.isChecked():
            pill.click()
            qapp.processEvents()
        assert pill.isChecked()
        page.run_bar.startButton.click()
        _stitch_wait(page, qapp)
        target = page._owner.prepared.request.module.output.target
        view = page.scientific
        _enable(qapp, view)
        from xrd_tools.core.angular_support import REASON_TEXT, SupportOrigin, SupportReason
        _wait(qapp, lambda: bool(_curves(view)))
        (_data, profile), = _curves(view).values()
        assert profile.available and profile.provenance['backend'] == 'multigeometry'
        # Diagnostic: Coverage (pixel counts) is not angular coverage
        page._diagnostic_combo.setCurrentText('Coverage')
        view.plot_options_dialog.accept_button.click()
        _wait(qapp, lambda: REASON_TEXT[SupportReason.DIAGNOSTIC_TRACE] in view.angular_support_message())
        assert not _curves(view)
        page._diagnostic_combo.setCurrentText('Intensity')
        view.plot_options_dialog.accept_button.click()
        _wait(qapp, lambda: bool(_curves(view)))
        assert page.shutdown().status is PageCleanup.CLEAN
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()
    # the saved result reopens with its saved support, through Stitch and through Int Browse, original inputs hidden.
    # Test-only oracle repair: a stitched record reads only at the path it records ("private candidate path",
    # analysis_artifact.py:2192 at cf49f3cb), so the result reopens where it was written, its source moved away.
    copy = target
    shutil.move(str(tmp_path / 'source'), str(tmp_path / 'hidden'))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(copy)
        _stitch_wait(page, qapp)
        view = page.scientific
        _enable(qapp, view)
        _wait(qapp, lambda: bool(_curves(view)))
        (_data, saved), = _curves(view).values()
        assert saved.origin is SupportOrigin.SAVED
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()
    with _page(qapp, _masked_series(tmp_path / 'int')) as (int_page, int_view, _observed):
        _enable(qapp, int_view)
        _reopen(qapp, int_page, copy)
        _wait(qapp, lambda: bool(_curves(int_view)))
        (_data, through_int), = _curves(int_view).values()
        assert through_int.origin is SupportOrigin.SAVED
        np.testing.assert_array_equal(through_int.supported_deg, saved.supported_deg)
