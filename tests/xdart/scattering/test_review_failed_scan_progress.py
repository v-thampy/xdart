"""Progress must include frames retained while a source-failed scan settles."""
import pytest

from tests.xdart.scattering.test_live_never_waits import (
    _output, test_a_container_frame_that_cannot_be_read_stops_its_scan_and_the_run_goes_on as _corrupt_container,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor


@pytest.mark.parametrize("live", [True, False])
def test_retained_container_frames_stay_in_the_run_total(tmp_path, caplog, monkeypatch, live):
    actual_settle = StandardRunExecutor._settle_failed_scan
    observed = []

    def settle(executor, run, target):
        result = actual_settle(executor, run, target)
        observed.append(run)
        return result

    monkeypatch.setattr(StandardRunExecutor, "_settle_failed_scan", settle)
    _corrupt_container(tmp_path, caplog, live)
    assert len(observed) == 1
    run = observed[0]
    print("FAILED-SCAN-PROGRESS", live, run.completed, run.total)
    kept = len(_output(tmp_path, "bad")[0])   # the failed scan's committed frames, as many as its epochs reached
    assert run.completed == run.total == kept + 3, (run.completed, run.total, kept)
