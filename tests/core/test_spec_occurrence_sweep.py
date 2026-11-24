"""SPEC occurrence checks share the byte observation within each transaction sweep."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.core.test_spec_occurrence import LATER_SCAN, _append, _rewrite_first_after
from tests.core.test_stitch_operation import _native_form
from xrd_tools.analysis import module_transaction
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.analysis.stitch_values import prepare_stitch_tool


@pytest.mark.parametrize("scans, change", [("1", "growth"), ("[1,2]", "growth"), ("[1,2]", "selected rewrite")])
def test_changed_spec_file_is_read_once_per_stitch_custody_sweep(tmp_path, monkeypatch, scans, change):
    form = _native_form(tmp_path, "multigeometry", scans=scans)
    spec = Path(form.spec_path)
    request = prepare_stitch_tool(form).request
    # A later scan and megabytes of unrelated data should never need to be read.
    _append(spec, LATER_SCAN.replace("#S 2", "#S 4") + "14 -10\n" * 400_000)

    passes = []
    descriptors = {}
    check_level = 0
    real_check = module_transaction._source_requalification
    real_open, real_read, real_close = os.open, os.read, os.close

    def check(*args, **kwargs):
        nonlocal check_level
        if check_level == 0:
            passes.append([])
            if change == "selected rewrite":
                _rewrite_first_after(spec, "#S 2", "15 -10", "15 -11")
        check_level += 1
        try:
            return real_check(*args, **kwargs)
        finally:
            check_level -= 1

    def observing_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if check_level and os.fspath(path) == str(spec):
            counts = passes[-1]
            counts.append(0)
            descriptors[descriptor] = counts, len(counts) - 1
        return descriptor

    def observing_read(descriptor, size):
        data = real_read(descriptor, size)
        if descriptor in descriptors:
            counts, index = descriptors[descriptor]
            counts[index] += len(data)
        return data

    def observing_close(descriptor):
        descriptors.pop(descriptor, None)
        return real_close(descriptor)

    monkeypatch.setattr(module_transaction, "_source_requalification", check)
    monkeypatch.setattr(os, "open", observing_open)
    monkeypatch.setattr(os, "read", observing_read)
    monkeypatch.setattr(os, "close", observing_close)
    result = run_stitch_operation(request)
    print(f"{scans} / {change}: bytes per verification pass = {passes}")
    if change == "selected rewrite":
        assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal
        assert not Path(request.module.output.target).exists()
        assert [len(sweep) for sweep in passes] == [1]
    else:
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        assert [len(sweep) for sweep in passes] == [2, 2]
    assert 0 < max(read for sweep in passes for read in sweep) <= 1 << 17


def test_rsm_bound_ub_and_same_file_members_share_the_authority_byte_check(tmp_path, monkeypatch):
    from dataclasses import replace

    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xrd_tools.analysis.rsm_operation import RSMUBOverride, _revalidate_rsm_v2_authorities
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.io.spec import spec_capture_scope, spec_file_view

    first, second = _write_member(tmp_path, 0), _write_member(tmp_path, 1)
    spec = Path(first.spec_path)
    second_scan = "#S 2 " + Path(second.spec_path).read_text().split("#S 1 ", 1)[1]
    _append(spec, "\n" + second_scan)
    form = replace(_form(tmp_path, (first, replace(second, spec_path=spec, scan="2.1"))),
                   ub=RSMUBOverride(spec_path=str(spec), scan="2.1"))
    request = prepare_rsm_tool_v2(form).request
    _append(spec, LATER_SCAN.replace("#S 2", "#S 3"))

    opened = []
    real_open = os.open

    def observing_open(path, flags, *args, **kwargs):
        if os.fspath(path) == str(spec):
            opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    with spec_capture_scope():
        spec_file_view(spec)
        monkeypatch.setattr(os, "open", observing_open)
        _revalidate_rsm_v2_authorities(request, cancel_token=None)
    print(f"two same-file RSM members and their UB authority: {len(opened)} verification passes")
    assert len(opened) == 1


def test_a_long_later_scan_header_needs_only_its_structural_marker(tmp_path, monkeypatch):
    from tests.core.test_spec_occurrence import _proof
    from xrd_tools.io.metadata import spec_occurrence_holds

    path = tmp_path / "growing"
    path.write_bytes(b"#F sample\n#S 1 test\n#N 2\n#L eta  I0\n0 10\n")
    proof = _proof(path, "1.1")
    _append(path, "#S 2 " + "x" * (2 << 20) + "\n")
    real_read = os.read
    read_bytes = 0

    def count_read(descriptor, size):
        nonlocal read_bytes
        values = real_read(descriptor, size)
        read_bytes += len(values)
        return values

    monkeypatch.setattr(os, "read", count_read)
    assert spec_occurrence_holds(proof)
    print(f"later giant #S header: {read_bytes} bytes read")
    assert 0 < read_bytes <= 1 << 17
