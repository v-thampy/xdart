"""Real owner/paint regressions found in BROWSE-CLICK-128 terminal review."""

import logging
import threading

import pytest

from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from tests.xdart.scattering.test_browse_click_128 import (
    _change_revision, _context, _perf_lines, _recheck, qapp,
)
from tests.xdart.scattering.test_browse_reopen_cleanup_pending import (
    _click_scan, _close, _int_page, _shows,
)
from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane


def test_changed_recheck_keeps_frame_through_real_pending_read(tmp_path, monkeypatch, qapp):
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    entered, release = threading.Event(), threading.Event()
    real_read = Browse1DHydrationLane._open_and_read

    def held_read(self, task, labels, fact_labels):
        entered.set()
        assert release.wait(20)
        return real_read(self, task, labels, fact_labels)

    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        context = _context(page)
        monkeypatch.setattr(Browse1DHydrationLane, "_open_and_read", held_read)
        _click_frame(qapp, page._shell, 1, settle=False)
        _wait(lambda: entered.is_set() or None)
        assert page._context_controller.navigation.current.local_frame_label == 2
        _change_revision(first, "mtime")
        _recheck(page, first)
        _wait(lambda: page._deferred_browse_reopen)
        release.set()
        _wait(lambda: _context(page) is not context and _shows(page, first))
        assert page._context_controller.navigation.current.local_frame_label == 2
    finally:
        release.set()
        _close(page, qapp)


def test_image_timing_waits_for_the_actual_scientific_reconcile(tmp_path, monkeypatch, qapp, caplog):
    monkeypatch.setenv("XDART_PERF", "1")
    caplog.set_level(logging.INFO)
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    entered, release = threading.Event(), threading.Event()
    real_read = Browse1DHydrationLane._open_and_read
    real_reconcile = page._shell.scientific.reconcile
    painted = []

    def held_read(self, task, labels, fact_labels):
        entered.set()
        assert release.wait(20)
        return real_read(self, task, labels, fact_labels)

    def observed_reconcile(science, *args, **kwargs):
        result = real_reconcile(science, *args, **kwargs)
        painted.append(science)
        return result

    try:
        monkeypatch.setattr(Browse1DHydrationLane, "_open_and_read", held_read)
        monkeypatch.setattr(page._shell.scientific, "reconcile", observed_reconcile)
        _click_scan(page, first)
        _wait(lambda: entered.is_set() or None)
        _wait(lambda: page._context_controller.request_current_browse_preview())
        page._refresh_shell()
        assert not painted or painted[-1].heavy is None
        assert not [line for line in _perf_lines(caplog) if "click→image=" in line]
        release.set()
        _wait(lambda: _shows(page, first, 1))
        _wait(lambda: any("click→image=" in line for line in _perf_lines(caplog)) or None)
        assert painted[-1].heavy is not None
        assert len([line for line in _perf_lines(caplog) if "click→image=" in line]) == 1
    finally:
        release.set()
        _close(page, qapp)


@pytest.mark.parametrize("failure", ("missing", "capture"))
def test_failed_recheck_keeps_the_shown_context(tmp_path, monkeypatch, qapp, failure):
    monkeypatch.setenv("XDART_PERF", "1")
    page, (first,) = _int_page(tmp_path, monkeypatch, ("first",))
    try:
        _click_scan(page, first)
        _wait(lambda: _shows(page, first, 1))
        context = _context(page)
        current = page._context_controller.navigation.current
        if failure == "missing":
            first.rename(first.with_suffix(".moved"))
        else:
            def failed_capture(path):
                raise OSError("revision unavailable")
            monkeypatch.setattr(loader_module, "capture_target_revision", failed_capture)
        _recheck(page, first)
        _wait(lambda: not page._context_controller.browse_pending or None)
        assert _context(page) is context and not context.released and not context.invalidated
        assert page._context_controller.navigation.current is current
        assert page._click_perf is None
    finally:
        _close(page, qapp)
