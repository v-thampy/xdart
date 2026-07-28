"""O-3N.R — the reviewer's malformed-input probes (handoff §15.5 R0).

RESHAPED for O-3N.R.1 §16.6.  The earlier revision answered filesystem
questions -- ``Path.is_file()``, an output stat, ``check_output_not_source()``
-- inside GUI admission, which is exactly the boundary §16.6 rejects: on a
beamline network path those "cheap" stats are not cheap, and filesystem,
link/collision, HDF5 and raw-stack facts belong to the worker.  So:

* PURE failures (blank Entry, non-constructible calibration) stay zero-delta
  refusals before carrier publication;
* FILESYSTEM/HDF5 failures visibly refuse IN THE WORKER, touch no source or
  output content, and leave the prior target byte-for-byte.

The rows below therefore assert the worker refusal and the untouched
filesystem, not a zero-delta GUI Start.

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


def test_nonexistent_nexus_source_refuses_in_the_worker(
        widget, tmp_path, monkeypatch):
    """§16.6: a filesystem fact, so the WORKER refuses -- visibly, with no
    source or output content touched."""
    _src, _out = _arm_nexus(_select_nexus(widget), tmp_path, entry="entry")
    wrangler = widget.wrangler
    missing = tmp_path / "does-not-exist.nxs"
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(missing))
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said = []
    thread.showLabel.connect(said.append)

    thread.run()

    assert any("refused" in text.lower() for text in said), said
    assert list(_out.iterdir()) == []


def test_file_shaped_output_refuses_in_the_worker(
        widget, tmp_path, monkeypatch):
    """§16.6: also a filesystem fact.  The file that occupies the output path
    keeps its bytes."""
    wrangler = _select_nexus(widget)
    _src, _out = _arm_nexus(wrangler, tmp_path, entry="entry")
    invalid_output = tmp_path / "not-a-directory"
    invalid_output.write_text("file")
    wrangler.parameters.child("Output", "h5_dir").setValue(
        str(invalid_output))
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said = []
    thread.showLabel.connect(said.append)

    thread.run()

    assert any("refused" in text.lower() for text in said), said
    assert invalid_output.read_text() == "file"


def test_blank_entry_is_not_invented_after_admission(widget, tmp_path):
    wrangler = _select_nexus(widget)
    _arm_nexus(wrangler, tmp_path, entry="")

    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen.source.entry == ""


def test_source_and_output_must_not_resolve_to_same_file(
        widget, tmp_path, monkeypatch):
    """§16.6/§16.7 item 10: the collision guard is the WORKER's, and the raw
    acquisition keeps its bytes."""
    wrangler = _select_nexus(widget)
    source, _out = _arm_nexus(
        wrangler, tmp_path, name="collision.nxs", entry="entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(source.parent))
    _start_recorder(wrangler, monkeypatch)
    before = source.read_bytes()
    wrangler.start()
    thread = wrangler.thread
    said = []
    thread.showLabel.connect(said.append)

    thread.run()

    assert any("refused" in text.lower() for text in said), said
    assert source.read_bytes() == before


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
