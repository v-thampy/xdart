"""Byte-lock gate for the multi-result per-mode NeXus layout (ADR-0003).

The committed fixture is the content signature of a deterministic multi-mode
scan.  Any change to the nested-subgroup names, the per-mode stacks, or the
primary_mode / multi_result_modes attrs (the frozen-once-shipped surface) fails
here.  Pure-io: no xdart needed (the writer is headless).
"""
import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "multimode_signature.json"


def test_multimode_record_content_locked(tmp_path):
    from tests.core._multimode_fixture import write_reference_multimode_scan
    from tests.core.h5sig import h5_content_signature

    out = write_reference_multimode_scan(str(tmp_path / "mm.nxs"))
    now = h5_content_signature(out)
    ref = json.loads(FIXTURE.read_text())

    missing = sorted(set(ref) - set(now))
    added = sorted(set(now) - set(ref))
    assert not missing and not added, (
        f"tree changed: missing={missing[:6]} added={added[:6]}"
    )
    diffs = [k for k in sorted(ref) if ref[k] != now[k]]
    assert diffs == [], (
        "content changed at: " + ", ".join(diffs[:8]) + "\n"
        + "\n".join(f"  {k}: ref={ref[k]} now={now[k]}" for k in diffs[:3])
    )
