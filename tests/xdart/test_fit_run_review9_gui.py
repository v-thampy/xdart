"""R9: terminal display and partial-result retention through real fits and the run owner."""
import threading

from tests.xdart.test_fit_run_controls_gui import _close, _loaded_host, _until, qapp  # noqa: F401


def test_final_read_failure_still_shows_the_last_fitted_frame(qapp):
    gates = [threading.Event() for _ in range(2)]
    host = _loaded_host(qapp, 2, gates, fail=(1,))
    dialog = host.dialog
    try:
        dialog.batch_btn.click()
        for gate in gates:
            gate.set()
        # Finish the real fits before processing GUI progress, so the last fitted frame is pending at terminal.
        worker = host.owner.worker._worker
        assert worker is not None
        worker.join(20)
        assert not worker.is_alive()
        assert _until(lambda: not host.computing)
        assert len(host.outcomes) == 2 and host.outcomes[0].ok and not host.outcomes[1].ok
        assert dialog.fit_view.title.text() == "frame0"
        assert any(series.role == "fit" for series in dialog.fit_view.projection.upper.series)
        assert dialog.table.rowCount() > 0
        assert len(dialog.trend.table()[0]) == 1
    finally:
        _close(host)


def test_stop_keeps_completed_nonconverged_fits_and_their_parameter_rows(qapp):
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)
    dialog = host.dialog
    try:
        dialog.adv_maxfev.setValue(1)  # real solver completion with best-fit arrays, success=False
        dialog.batch_btn.click()
        gates[0].set()
        assert _until(lambda: dialog.fit_view.title.text() == "frame0")
        assert host._shown_outcome.result.fit_success is False
        assert host._shown_outcome.params and len(dialog.trend.table()[0]) == 1
        assert _until(lambda: loaded == [0, 1])
        dialog.batch_btn.click()
        gates[1].set()
        assert _until(lambda: not host.computing)
        assert loaded == [0, 1]
        assert len(host.outcomes) == 2 and all(not outcome.ok and outcome.params for outcome in host.outcomes)
        assert host.holds_batch and len(dialog.trend.table()[0]) == 2
        assert dialog.fit_view.title.text() == "frame1"
        assert "Stopped before any frame" not in dialog.status.text()
    finally:
        _close(host)
