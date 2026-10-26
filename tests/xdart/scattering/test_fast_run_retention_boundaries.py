"""Finished 1-D copies preserve a reopened scan's prefix and include support in their memory budget."""
import numpy as np

from tests.xdart.scattering.test_fast_run_overlay_keeps_every_curve import (
    _intent, _overlay, _starved_until, _ended_on_browse, _drawn,
)
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_p2_0_light_1d_cutover import _terminal_run
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.events import CleanupStatus


def test_a_reopened_live_scan_keeps_its_unpainted_prefix_after_it_retires_again(qapp, tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    intent = _intent(tmp_path, live=True)
    intent.output_mode = 'Append'
    with _page(qapp, intent) as (page, _view, _events):
        _overlay(page)
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
        for processed, (scan, label) in enumerate((('a', 1), ('b', 1), ('a', 2), ('c', 1)), 1):
            _frame(raw, scan, label, 'txt', float(processed))
            assert _starved_until(lambda: run.files_processed >= processed), (scan, run.files_processed)
        retained_a = sorted(frame.local_frame_label for frame in run.display._released_1d
                            if frame.source_scan == 'a')
        assert retained_a == [1, 2], 'retiring the resumed output discarded its retained prefix'
        page._shell.run_controls.stopButton.click()
        assert _starved_until(lambda: run.terminal_emitted)
        assert _ended_on_browse(qapp, page, 'c'), (page._notice_text, _drawn(page))
        assert _drawn(page) == ['a_int1d.nexus', 'a_int1d.nexus', 'b_int1d.nexus', 'c_int1d.nexus']


def test_released_curve_budget_counts_angular_support(tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import browse_1d_display
    executor, identity, run, owner, _target, _terminal = _terminal_run(
        tmp_path, processing_mode='Int 2D', record_angular_support=True)
    view, arrays = None, ()
    try:
        view = owner.publications.get(0).view
        support = view.angular_support
        assert support is not None and support.available
        arrays = (view.axis_1d.values, view.intensity_1d, view.sigma_1d)
        main = sum(np.asarray(array).nbytes for array in arrays if array is not None)
        limit = main + support.supported_deg.nbytes // 2
        view, arrays = None, ()  # release the test's own lease-backed borrow before closing
        monkeypatch.setattr(browse_1d_display, 'MAX_BROWSE_1D_DETACHED_BYTES', limit)
        run.display.keep_released_1d(owner)
        actual = 0
        for payload in run.display._released_1d.values():
            kept = payload.view
            actual += sum(np.asarray(array).nbytes for array in (
                kept.axis_1d.values, kept.intensity_1d, kept.sigma_1d,
                kept.angular_support.supported_deg) if array is not None)
        assert actual <= limit, 'the retained support vector exceeds the declared detached cap'
    finally:
        view, arrays = None, ()
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED


def test_released_curves_keep_no_detector_or_two_dimensional_arrays(tmp_path):
    executor, identity, run, owner, _target, _terminal = _terminal_run(
        tmp_path, processing_mode='Int 2D', record_angular_support=True)
    try:
        assert owner.publications.get(0).view.has_2d
        run.display.keep_released_1d(owner)
        [payload] = run.display._released_1d.values()
        kept = payload.view
        assert kept.has_1d and kept.angular_support.available
        assert kept.raw is None and kept.thumbnail is None
        assert kept.intensity_2d is None and kept.sigma_2d is None
        assert kept.axis_2d_x is None and kept.axis_2d_y is None
        assert not payload.extra_views_2d
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
