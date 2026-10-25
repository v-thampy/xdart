"""A dormant saved output has no session whose first energy grid can be retained."""
from dataclasses import replace

import fabio
import numpy as np

from tests.xdart.scattering.test_energy_scan_append_grid import _frame, _stack, _STOP
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _labels, _said, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.reduction import FrameBackgroundPlan


def test_a_dormant_saved_energy_scan_still_stops_before_a_wider_session_is_opened(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    bg = tmp_path / "background.tif"
    fabio.tifimage.TifImage(data=np.zeros((195, 487), dtype=np.uint16)).write(str(bg))
    background = FrameBackgroundPlan(mode="Single BG File", locator=str(bg))

    def intent(live):
        return replace(_intent(tmp_path, raw, live=live), background=background)

    for index, value in enumerate((16., 16.1, 16.2), 1):
        _frame(raw, "grow", index, value)
    executor, identity, events = _run_to_terminal(intent(False), request_value=9881)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    before = (tmp_path / "processed" / "grow_int1d.nexus").read_bytes()
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, intent(True), request_value=9882)
    events = []
    try:
        _wait(executor, events, lambda values: bool(executor.processed_live_revisions(identity)) or any(
            event.kind in _TERMINAL for event in values), 20.)
        output = executor._exact_run(identity).output
        assert output.owns_target(tmp_path / "processed" / "grow_int1d.nexus") and output.session is None
        _frame(raw, "grow", 4, 16.3)
        _wait(executor, events, lambda values: _said(values, "grow_int1d.nexus stopped growing") or any(
            event.kind in _TERMINAL for event in values), 20.)
        _frame(raw, "next", 1, 16.)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or any(
            event.kind in _TERMINAL for event in values), 20.)
    finally:
        terminal = _stop(executor, identity, events)
    print("DORMANT-ENERGY", terminal)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _said(events, "grow_int1d.nexus stopped growing")
    assert _stack(tmp_path)[0] == (1, 2, 3) and _stack(tmp_path, "next")[0] == (1,)
    assert (tmp_path / "processed" / "grow_int1d.nexus").read_bytes() == before
