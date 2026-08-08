# -*- coding: utf-8 -*-
"""The one session cadence boundary table used by the C3 cutover."""
from __future__ import annotations


def test_session_bound_cadence_boundary(tmp_path, monkeypatch):
    from tests.xdart.test_vnext_p0_c3_dynamic_gui_mount import (
        _c3_real_source_case, _c3_run,
    )

    for mode, interval in (("Int 2D", 8), ("Int 1D", 1000)):
        with monkeypatch.context() as scoped:
            adapter = None
            case = _c3_real_source_case(
                tmp_path / f"cadence-{interval}", "tiff", processing_mode=mode,
            )
            try:
                _trace, adapters = _c3_run(case, scoped)
                assert len(adapters) == 1
                adapter = adapters[0]
                policy = adapter._session.policy
                assert policy is adapter._policy
                assert policy.allocation is case.worker.publication_store.allocation
                assert policy.flush.cap == policy.allocation.staging_items
                assert policy.flush.margin == 8
                assert policy.flush.interval == interval
                threshold = policy.flush.hard_threshold()
                cases = (
                    (interval - 1, 0, False, False),
                    (interval, 0, False, True),
                    (1, threshold - 1, False, False),
                    (1, threshold, False, True),
                    (0, 0, True, False),
                    (1, 1, True, True),
                )
                for frames, pressure, force, expected in cases:
                    assert policy.should_flush(
                        frames_since_flush=frames,
                        unsaved_in_memory=pressure,
                        force=force,
                    ) is expected
            finally:
                case.worker.command = "start"
                case.worker._close_reduction_session()
                retained = getattr(
                    case.worker, "_retained_scan_session_adapter", None,
                )
                if retained is not None:
                    if adapter is not None:
                        assert retained is adapter
                    assert retained.release_retained_custody() is True
