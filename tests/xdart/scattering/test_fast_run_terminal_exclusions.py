"""A real Run's terminal need not wait for frames its Overlay correctly refuses."""
from tests.xdart.scattering.test_fast_run_overlay_keeps_every_curve import _intent, _overlay, _starved_until
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.state_machine import RunPhase


def test_terminal_releases_when_selected_frames_are_drawn_or_refused(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import page as page_owner
    monkeypatch.setattr(page_owner, '_TERMINAL_PAINT_STALL_S', 60.)
    monkeypatch.setattr(page_owner, '_TERMINAL_PAINT_HOLD_S', 120.)
    # Two scans, so the first is released while the view is starved and its kept curve awaits the end (the end's
    # Browse shows the second itself). Each is reduced on its own q range, 16 and 17 keV: in q their grids differ, so
    # the first is refused (in 2theta both come back to the detector's own extent).
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'a', 1, 'txt', 1.0)
    _frame(raw, 'b', 1, 'txt', 2.0)
    sidecar = raw / 'b_0001.txt'
    sidecar.write_text(sidecar.read_text().replace('energy = 16', 'energy = 17'))
    with _page(qapp, _intent(tmp_path, live=False)) as (page, view, _events):
        _overlay(page)
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        run = page._run_executor._exact_run(page._lifecycle.active_run_identity)
        assert _starved_until(lambda: run.terminal_emitted)
        page._drain_executor()
        assert page._terminal_hold is not None
        assert view.excluded_trace_keys, 'the two energies sample incompatible q grids'
        assert not page._context_controller.unpainted_trace_frames()
        page._drain_executor()
        assert page._terminal_hold is None
        assert page._lifecycle.phase is RunPhase.IDLE
