"""Direct real executor regression probe for Claude F2; no product doubles."""
from threading import Event

import h5py
import numpy as np
import pytest
import tifffile

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import (
    _drain_until, _intent, _run_to_terminal, _start, _TERMINAL, _written,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus


@pytest.mark.parametrize("mask_enabled", (False, True))
def test_real_stop_append_matches_one_pass(tmp_path, monkeypatch, mask_enabled):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "scan_0001.tif"
    for label in range(1, 17):
        data = np.full((195, 487), 100, dtype=np.uint32)
        data[97, 200] = np.iinfo(np.uint32).max if label == 1 else 1000000
        tifffile.imwrite(raw_root / f"scan_{label:04}.tif", data)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    full_target = tmp_path / "full" / "output.nexus"
    resume_target = tmp_path / "resumed" / "output.nexus"
    full_target.parent.mkdir()
    resume_target.parent.mkdir()

    def intent(target, mode="Overwrite"):
        value = _intent(raw, target, poni, output_mode=mode, npt=32)
        value.threshold.mask_saturation = mask_enabled
        return value

    def arrays(target):
        with h5py.File(_written(target), "r") as handle:
            group = handle["entry/integrated_1d"]
            return group["frame_index"][()], group["intensity"][()]

    full, full_id, full_events = _run_to_terminal(intent(full_target), request_value=9101)
    try:
        assert next(event for event in full_events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    finally:
        assert full.close(full_id).cleanup_status is CleanupStatus.CLEANED
    full_labels, expected = arrays(full_target)

    entered, release = Event(), Event()
    original_submit = DynamicOutputAdapter.submit

    def paced_submit(adapter, frame, *args, **kwargs):
        if int(frame.index) == 9:
            entered.set()
            assert release.wait(30.0), "source pacing was not released"
        return original_submit(adapter, frame, *args, **kwargs)

    stopped = StandardRunExecutor(join_timeout=2.0)
    stop_id = None
    with monkeypatch.context() as scoped:
        scoped.setattr(DynamicOutputAdapter, "submit", paced_submit)
        try:
            stop_id = _start(stopped, intent(resume_target), request_value=9102)
            assert entered.wait(30.0)
            stopped.stop(stop_id)
            release.set()
            stop_events = _drain_until(stopped, lambda values: any(event.kind in _TERMINAL for event in values))
            terminal = next(event for event in stop_events if event.kind in _TERMINAL)
            assert terminal.kind is StandardEventKind.STOPPED, terminal
            assert 0 < terminal.completed < 16
        finally:
            release.set()
            if stop_id is not None:
                assert stopped.close(stop_id).cleanup_status is CleanupStatus.CLEANED
    prefix_labels, prefix = arrays(resume_target)
    np.testing.assert_array_equal(prefix, expected[:len(prefix)])

    appended, append_id, append_events = _run_to_terminal(intent(resume_target, "Append"), request_value=9103)
    try:
        assert next(event for event in append_events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    finally:
        assert appended.close(append_id).cleanup_status is CleanupStatus.CLEANED
    labels, actual = arrays(resume_target)
    np.testing.assert_array_equal(labels, full_labels)
    np.testing.assert_array_equal(actual[:len(prefix)], prefix)
    delta = np.abs(actual[len(prefix):] - expected[len(prefix):])
    print("SATURATION_APPEND", "enabled", mask_enabled, "prefix", prefix_labels.tolist(),
          "different_values", np.count_nonzero(delta), "max_abs", float(delta.max()), flush=True)
    np.testing.assert_array_equal(actual[len(prefix):], expected[len(prefix):])
