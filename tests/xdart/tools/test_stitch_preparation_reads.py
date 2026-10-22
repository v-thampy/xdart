"""PERF-STITCH-1 (Codex, 2026-09-30): one Stitch check reads each source's metadata table and each frame's
layout once, and passes those admitted facts through the receipt, the manifest and the request; the source
checks stay at the operation's boundary (a source that changes after the check refuses the run)."""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis import scan_operations, stitch_native
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from tests.xdart.tools.test_stitch_native_values import _native_form


def _counting(monkeypatch):
    calls = Counter()
    table, facts = scan_operations.run_metadata_table, stitch_native._frame_facts

    def counted_table(*args, **kwargs):
        calls["metadata table"] += 1
        return table(*args, **kwargs)

    def counted_facts(*args, **kwargs):
        calls["frame facts"] += 1
        return facts(*args, **kwargs)
    for module in list(sys.modules.values()):
        if getattr(module, "run_metadata_table", None) is table:
            monkeypatch.setattr(module, "run_metadata_table", counted_table)
    monkeypatch.setattr(stitch_native, "_frame_facts", counted_facts)
    return calls


def test_one_check_reads_each_table_and_each_frames_layout_once(tmp_path, monkeypatch):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    calls = _counting(monkeypatch)
    prepared = prepare_stitch_tool(form)
    assert len(prepared.requests) == 1
    assert calls == {"metadata table": 1, "frame facts": 1}  # the parent: 6 and 3
    request, = prepared.requests
    # the frame list the page shows is the one this check bound, equal to a fresh read
    fresh = stitch_native._validated_native_members(
        request, stitch_native.frame_seeds(request.module.source, request.manifest, request.plan))
    assert prepared.preview_members[0] == tuple(fresh)


def test_a_source_changed_after_the_check_refuses_the_run(tmp_path):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    request, = prepare_stitch_tool(form).requests
    image = next(Path(form.image_dir).glob("sample_scan1_0000*"))
    frame = np.fromfile(image, dtype=np.int32)
    time.sleep(.01)
    (frame + 1).tofile(image)  # another revision after the check
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal
