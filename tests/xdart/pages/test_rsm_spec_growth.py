"""NEXT-1219-SPEC-OCCURRENCE S8: an RSM Run on a finished scan is saved while the next scan is being written.

His word (2026-11-23): "Bind each scan's bytes (Recommended)": at a beamline a SPEC file grows every count time;
a Run on scan 1 must not be refused because scan 2 is being recorded in the same file. Real page, owner, worker
and gridding; the later scan lands on the owner's worker at the Run's authority checks.
"""
from __future__ import annotations

import threading
from pathlib import Path

from tests.xdart.pages.test_rsm_page import _close, _page, _synthetic_source, _wait, _with_source, qapp  # noqa: F401


def test_an_rsm_run_is_saved_while_a_later_scan_is_written_to_its_spec_file(qapp, tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        spec = _synthetic_source(tmp_path, full=True)
        _with_source(page, tmp_path, scans="1")
        output = Path(page.build_form().output_path)
        checks = rsm_operation._revalidate_rsm_v2_authorities
        landed = []

        def growing_check(*args, **kwargs):
            with spec.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(f"\n#S {3 + len(landed)} ascan eta 0 1 1 1\n#L eta  Seconds\n0 1\n")
            landed.append((threading.current_thread() is not threading.main_thread(), page._owner.busy))
            return checks(*args, **kwargs)

        monkeypatch.setattr(rsm_operation, "_revalidate_rsm_v2_authorities", growing_check)
        page.begin_run()
        _wait(page, qapp, lambda: page.shown is not None and not page.active(), seconds=60.0)
        assert landed and all(worker and busy for worker, busy in landed), landed  # while the owner ran it
        assert output.is_file() and label.full_text() == f"Saved {output.name}", label.full_text()
        assert Path(page.browser.current_result()).name == output.name
    finally:
        _close(page, qapp)
