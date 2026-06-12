"""6a gate: the refactored writer produces a content-identical v2 record.

The committed fixture is the content signature of a deterministic scan
written by the PRE-6a (all-xdart) writer.  Any 6a step that changes what
lands on disk for an identical scan fails here.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("xdart", reason="gate exercises the GUI-side writer")

FIXTURE = Path(__file__).parent / "fixtures" / "v2_record_signature_pre6a.json"


def test_v2_record_content_identical_to_pre6a(tmp_path):
    from tests.core._v2_record_fixture import write_reference_scan
    from tests.core.h5sig import h5_content_signature

    out = write_reference_scan(str(tmp_path / "ref.nxs"),
                               str(tmp_path / "proj"))
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
