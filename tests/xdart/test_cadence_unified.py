# -*- coding: utf-8 -*-
"""The one session cadence boundary table used by the current GUI run owner."""
from __future__ import annotations


def test_session_bound_cadence_boundary(tmp_path, monkeypatch):
    from tests.xdart.scattering.test_p1b_output_graph import (
        _intent, _run_to_terminal, _write_tiff, write_poni,
    )
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.events import CleanupStatus

    for mode, interval in (("Int 2D", 8), ("Int 1D", 1000)):
        with monkeypatch.context() as scoped:
            root = tmp_path / f"cadence-{interval}"
            root.mkdir()
            raw, poni = root / "scan_0001.tif", root / "cal.poni"
            _write_tiff(raw, 1)
            write_poni(poni)
            intent = _intent(raw, root / "processed.nexus", poni,
                             processing_mode=mode)
            sessions = []
            real_open = dynamic_output.open_headless_scan_session

            def open_session(*args, **kwargs):
                session = real_open(*args, **kwargs)
                assert session.policy is kwargs["policy"]
                sessions.append(session)
                return session

            scoped.setattr(dynamic_output, "open_headless_scan_session", open_session)
            executor, identity, events = _run_to_terminal(
                intent, request_value=21000 + interval,
            )
            try:
                assert any(event.kind is StandardEventKind.FINISHED for event in events)
                session, = sessions
                policy = session.policy
                owner, = executor._exact_run(identity).display.artifacts.values()
                assert policy is session._policy
                assert policy.allocation is owner.publications.allocation
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
                assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
