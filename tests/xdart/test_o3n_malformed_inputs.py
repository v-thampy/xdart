"""O-3N.R — the reviewer's malformed-input probes (handoff §15.5 R0).

Promoted from ``/Users/vthampy/repos/tmp/test_codex_o3n_malformed.py``.  Every
row is a ZERO-DELTA refusal at the real ``nexusWrangler.start()``: a run that
cannot name a real input, a real output directory, or an entry must leave
carrier, ledger, thread, buttons, pending slot and generation floor exactly as
it found them, and must never reach the worker.

The helpers and fixtures come from the O-3N module so both oracles observe the
same production seam.
"""

from __future__ import annotations

from tests.xdart.test_o3n_nexus_freeze_identity import (  # noqa: F401
    _arm_nexus,
    _select_nexus,
    _start_recorder,
    _zero_delta_snapshot,
    qapp,
    widget,
)


def test_nonexistent_nexus_source_refuses_before_mutation(
        widget, tmp_path, monkeypatch):
    wrangler = _select_nexus(widget)
    _src, _out = _arm_nexus(wrangler, tmp_path, entry="entry")
    missing = tmp_path / "does-not-exist.nxs"
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(missing))
    events = _start_recorder(wrangler, monkeypatch)
    before = _zero_delta_snapshot(widget, wrangler)

    wrangler.start()

    assert events == []
    assert _zero_delta_snapshot(widget, wrangler) == before


def test_file_shaped_output_refuses_before_mutation(
        widget, tmp_path, monkeypatch):
    wrangler = _select_nexus(widget)
    _src, _out = _arm_nexus(wrangler, tmp_path, entry="entry")
    invalid_output = tmp_path / "not-a-directory"
    invalid_output.write_text("file")
    wrangler.parameters.child("Output", "h5_dir").setValue(
        str(invalid_output))
    events = _start_recorder(wrangler, monkeypatch)
    before = _zero_delta_snapshot(widget, wrangler)

    wrangler.start()

    assert events == []
    assert _zero_delta_snapshot(widget, wrangler) == before


def test_blank_entry_is_not_invented_after_admission(widget, tmp_path):
    wrangler = _select_nexus(widget)
    _arm_nexus(wrangler, tmp_path, entry="")

    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen.source.entry == ""


def test_source_and_output_must_not_resolve_to_same_file(
        widget, tmp_path, monkeypatch):
    wrangler = _select_nexus(widget)
    source, _out = _arm_nexus(
        wrangler, tmp_path, name="collision.nxs", entry="entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(source.parent))
    events = _start_recorder(wrangler, monkeypatch)
    before = _zero_delta_snapshot(widget, wrangler)

    wrangler.start()

    assert events == []
    assert _zero_delta_snapshot(widget, wrangler) == before


def test_worker_scan_owns_the_frozen_output_before_writer_use(
        widget, tmp_path, monkeypatch):
    wrangler = _select_nexus(widget)
    source, output = _arm_nexus(
        wrangler, tmp_path, name="source.nxs", entry="entry")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread

    scan = thread._initialize_scan(source.stem)

    assert scan.data_file == str(output / "source.nxs")
