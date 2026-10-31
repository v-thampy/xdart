"""PERF-BROWSE-OPEN-128 O6 (real data): a large output's open through the real Browse loader (Claude, 2026-10-30).

His words: "why does it take over 2s to load a nexus file in the data browser? can you see if this can be sped up"
(his case: "Large burst files"). Measured on dev before stage A: the 2500-frame, 1.47 GB bo_0 output spent 0.73 s in
the loader's initial seal (a SHA-256 of the whole file) and 1.45 s in its catalog. Stage A seals the file revision
instead. Timing comes from the loader's own BrowseLoadTiming (perf_enabled); targets are guidelines, so only the seal
stage, which stage A removes, is held to a bound. Skipped without XDART_TEST_DATA: no real-data verdict then.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

BURST = "xdart_processed_data/bo_0_506V_0p1ms_sfpx_45p19_halpha_0p30_burst_00001_int2d.nexus"


def _load(path):
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest, BrowseLoadStatus
    loader = BrowseLoader(perf_enabled=lambda: True)
    request = BrowseLoadRequest("o6", 1, str(path))
    loader.begin(request)
    deadline = time.monotonic() + 60
    while (outcome := loader.poll(request)) is None:
        assert time.monotonic() < deadline, "the load did not finish"
        time.sleep(0.005)
    context = loader.consume(outcome)
    assert outcome.status is BrowseLoadStatus.READY, outcome.detail
    try:
        return outcome.timing
    finally:
        assert loader.release_context(context).cleanup_status.value == "cleaned"


def test_o6_a_2500_frame_open_spends_no_time_hashing_the_file():
    root = os.environ.get("XDART_TEST_DATA")
    path = Path(root or "", BURST)
    if not root or not path.is_file():
        pytest.skip("XDART_TEST_DATA with the bo_0 burst output is needed for this real-data row")
    timings = [_load(path.resolve()) for _ in range(2)]
    for timing in timings:
        print(f"O6 {path.name} records={timing.record_count} seal={timing.initial_seal_s:.3f}s "
              f"catalog={timing.record_iteration_s:.3f}s worker={timing.worker_total_s:.3f}s seal_mode={timing.seal_mode}")
        assert timing.seal_mode == "snapshot" and timing.record_count == 2500
        assert timing.initial_seal_s < 0.05, "the seal no longer reads and hashes the 1.47 GB file"
