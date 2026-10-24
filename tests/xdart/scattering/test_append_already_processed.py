"""A directory Append skips a scan it has already processed, before reading it again (maintainer, 2026-10-23: "If
something has already been processed, it should be skipped anyway, what's the need to check anything?"; Codex's
design review DESIGN-append-smb-125, with Astra).

Beamline 1.2.4 (Windows share, TIFF + .txt): eight completed scans were refused "same-extent source identity
changed: digest", a fingerprint of file bookkeeping, while every TIFF's name, size and time was unchanged, and every
completed scan was read again in full before its skip. Real directory Runs over real TIFF + .txt scans; work is
counted by delegating wrappers on the named seams."""
from __future__ import annotations

import os

import h5py
import pytest

import xrd_tools.io.record_writer as record_writer
from tests.xdart.scattering.test_live_revalidation_cost_run import PILATUS, TINY, _poni, _Sidecar, _tiff
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

_ENERGY = "Using the scan energy, 16.000 keV; the PONI says 12.398 keV."


def _intent(tmp_path, raw, *, mode="Append", npt=8, processing="Int 1D", shape=TINY):
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
        poni_file=str(_poni(tmp_path, shape)), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode=mode, processing_mode=processing, live_mode=False, max_cores=1,
        bai_1d_args={"npt": npt, "method": "numpy"}, bai_2d_args={"npt_rad": 8, "npt_azim": 8, "method": "numpy"})


def _scan(raw, stem, frames, start=0, shape=TINY):
    scan = _Sidecar(raw, shape, stem)
    for k in range(start, start + frames):
        scan.frame(k)
    return scan


def _run(tmp_path, raw, request_value, **intent):
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, **intent), request_value=request_value)
    catalog = executor.frame_catalog(identity)
    executor.close(identity)
    return events, catalog


def _shown(events, catalog):
    """What the page shows: each event's kind, counts, artifact and status line, the terminal's warnings, and the
    navigation rows."""
    rows = [(event.kind.name, event.completed, event.total, str(event.artifact or "").rsplit(os.sep, 1)[-1],
             event.detail, event.artifact_completed, event.artifact_total, event.files_processed,
             event.files_skipped, event.files_pending, event.files_discovered) for event in events]
    warnings = [warning for event in events if event.kind in _TERMINAL for warning in event.warnings]
    navigation = [(key.source_scan, key.local_frame_label, key.work_ordinal) for key in catalog.entries]
    return rows, warnings, navigation


def _labels(tmp_path, stem, dimension="1d"):
    with h5py.File(tmp_path / "processed" / f"{stem}_int{dimension}.nexus", "r") as handle:
        return tuple(int(value) for value in handle[f"entry/integrated_{dimension}/frame_index"][()])


class _Seams:
    """Delegating counters: which outputs were materialized (source capture, graph, Append qualification), given an
    Append preflight (whose admission hashes the whole output), or given a writer."""

    def __init__(self, monkeypatch):
        self.materialized, self.preflighted, self.writers = [], [], 0
        real_materialize = run_executor.materialize_deferred_output
        real_preflight = dynamic_output.prepare_append_preflight
        real_begin = record_writer.NexusRecordWriter.begin

        def materialize(receipt, session, entry, *args, **kwargs):
            self.materialized.append(entry.target.name)
            return real_materialize(receipt, session, entry, *args, **kwargs)

        def preflight(target, *args, **kwargs):
            self.preflighted.append(os.path.basename(str(target)))
            return real_preflight(target, *args, **kwargs)

        def begin(writer, *args, **kwargs):
            self.writers += 1
            return real_begin(writer, *args, **kwargs)

        monkeypatch.setattr(run_executor, "materialize_deferred_output", materialize)
        monkeypatch.setattr(dynamic_output, "prepare_append_preflight", preflight)
        monkeypatch.setattr(record_writer.NexusRecordWriter, "begin", begin)


def _completed_and_new(tmp_path, shape=TINY):
    raw = tmp_path / "raw"
    raw.mkdir()
    _scan(raw, "scana", 3, shape=shape)
    events, _catalog = _run(tmp_path, raw, 9101, shape=shape)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    _scan(raw, "scanb", 2, shape=shape)
    return raw


#: What 31f8725e showed for a completed scan beside a new one (its no-op path read scana again in full).
_PARENT_ROWS = [
    ("DISCOVERY", 0, 0, "", "Running · 0 processed · 0 skipped · 5 pending · 5 discovered", 0, 0, 0, 0, 5, 5),
    ("CONTEXT_READY", 0, 0, "scana_int1d.nexus", f"scana_int1d.nexus: {_ENERGY}", 0, 0, 0, 0, 0, 0),
    ("DISCOVERY", 3, 3, "scana_int1d.nexus", "Running · 0 processed · 0 skipped · 5 pending · 5 discovered",
     3, 3, 0, 0, 5, 5),
    ("DISCOVERY", 3, 3, "scana_int1d.nexus", "Running · 3 processed · 0 skipped · 2 pending · 5 discovered",
     3, 3, 3, 0, 2, 5),
    ("CONTEXT_READY", 0, 0, "scanb_int1d.nexus", "", 0, 0, 0, 0, 0, 0),
    ("DISCOVERY", 3, 5, "scanb_int1d.nexus", "Running · 3 processed · 0 skipped · 2 pending · 5 discovered",
     0, 2, 3, 0, 2, 5),
    ("FRAME_READY", 4, 5, "scanb_int1d.nexus", "Running · 3 processed · 0 skipped · 2 pending · 5 discovered",
     1, 2, 3, 0, 2, 5),
    ("FRAME_READY", 5, 5, "scanb_int1d.nexus", "Running · 3 processed · 0 skipped · 2 pending · 5 discovered",
     2, 2, 3, 0, 2, 5),
    ("DISCOVERY", 5, 5, "scanb_int1d.nexus", "Running · 5 processed · 0 skipped · 0 pending · 5 discovered",
     2, 2, 5, 0, 0, 5),
    ("FINISHED", 5, 5, "scanb_int1d.nexus", "Complete · 5 processed · 0 skipped · 0 pending · 5 discovered",
     2, 2, 5, 0, 0, 5),
]
_PARENT_NAVIGATION = [("scana", 1, 1), ("scana", 2, 2), ("scana", 3, 3), ("scanb", 1, 4), ("scanb", 2, 5)]


@pytest.mark.parametrize("touched", (False, True), ids=("unchanged", "metadata-rewritten"))
def test_a_completed_scan_is_skipped_as_before_without_reading_it_again(tmp_path, monkeypatch, touched):
    """The completed scan shows as its no-op showed (progress, counts, navigation, terminal), with no
    materialization, preflight hash, writer or change to its output, and no warning, also when its .txt files were
    rewritten (the beamline's digest-only drift). The one difference: the energy notice comes with the first scan a
    Run reads, not with a scan it no longer reads."""
    raw = _completed_and_new(tmp_path)
    output = tmp_path / "processed" / "scana_int1d.nexus"
    if touched:
        for k in range(3):
            (raw / f"scana_{k:04d}.txt").write_text(
                (raw / f"scana_{k:04d}.txt").read_text() + "# rewritten after processing\n")
    before = (output.read_bytes(), output.stat().st_mtime_ns)
    seams = _Seams(monkeypatch)
    events, catalog = _run(tmp_path, raw, 9102)
    rows, warnings, navigation = _shown(events, catalog)
    expected = [list(row) for row in _PARENT_ROWS]
    expected[1][4], expected[4][4] = "", f"scanb_int1d.nexus: {_ENERGY}"
    # Status contract change (maintainer 2026-10-23: "it says there are 24 images or so remaining and then goes
    # through them in a couple of seconds"): files admission already found processed count from the first status
    # on, and showing their scan adds nothing again.
    for row in (0, 2):
        expected[row][4] = "Running · 3 processed · 0 skipped · 2 pending · 5 discovered"
        expected[row][7], expected[row][9] = 3, 2
    assert rows == [tuple(row) for row in expected]
    assert warnings == [] and navigation == _PARENT_NAVIGATION
    assert seams.materialized == ["scanb_int1d.nexus"] and seams.preflighted == ["scanb_int1d.nexus"]
    assert seams.writers == 1 and (output.read_bytes(), output.stat().st_mtime_ns) == before
    assert _labels(tmp_path, "scana") == (1, 2, 3) and _labels(tmp_path, "scanb") == (1, 2)


def test_a_completed_scan_that_gained_frames_appends_them(tmp_path, monkeypatch):
    raw = _completed_and_new(tmp_path)
    _scan(raw, "scana", 1, start=3)
    seams = _Seams(monkeypatch)
    events, _catalog = _run(tmp_path, raw, 9103)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert sorted(seams.materialized) == ["scana_int1d.nexus", "scanb_int1d.nexus"]
    assert _labels(tmp_path, "scana") == (1, 2, 3, 4) and _labels(tmp_path, "scanb") == (1, 2)


def test_a_completed_scan_does_not_capture_its_source_before_skipping(tmp_path, monkeypatch):
    """The agreed early no-op includes admission, before deferred materialization."""
    from xrd_tools.sources.execution_graph import SourceFileState

    raw = _completed_and_new(tmp_path)
    captured = []
    real_capture = SourceFileState.capture

    def capture(cls, path):
        captured.append(os.path.basename(str(path)))
        return real_capture(path)

    monkeypatch.setattr(SourceFileState, "capture", classmethod(capture))
    events, _catalog = _run(tmp_path, raw, 9105)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert not any(name.startswith("scana_") for name in captured), captured


@pytest.mark.parametrize("change", ("renamed", "raw-rewritten", "settings", "modes"))
def test_what_is_not_already_processed_goes_the_full_way(tmp_path, monkeypatch, change):
    """Other files under the same count, a TIFF whose size or time changed, other processing settings or other modes
    are never taken for completion: the scan goes the full way and keeps its existing outcome."""
    raw = _completed_and_new(tmp_path)
    intent = {}
    if change == "renamed":
        for suffix in (".tif", ".txt"):
            (raw / f"scana_0002{suffix}").rename(raw / f"scana_0005{suffix}")
    elif change == "raw-rewritten":
        member = raw / "scana_0001.tif"
        state = member.stat()
        _tiff(member, 77, TINY)
        os.utime(member, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000_000))
    elif change == "settings":
        intent["npt"] = 16
    else:
        intent["processing"] = "Int 2D"
    seams = _Seams(monkeypatch)
    events, _catalog = _run(tmp_path, raw, 9104, **intent)
    assert "scana_int1d.nexus" in seams.materialized or "scana_int2d.nexus" in seams.materialized
    warnings = [warning for event in events if event.kind in _TERMINAL for warning in event.warnings]
    if change in ("renamed", "settings"):
        assert any("scana" in warning for warning in warnings), warnings
    elif change == "raw-rewritten":
        # Policy change (maintainer 2026-10-23, Codex N-0243a1bb): the same size, and its sidecar still reads the
        # acquisition time its frame recorded, so the rewritten TIFF counts as the same acquisition and is not
        # reduced again. The rule's accepted blind spot: new pixels under an unchanged sidecar time pass as unchanged.
        assert not any("scana" in warning for warning in warnings), warnings


def test_a_completed_scan_under_a_detector_ceiling_is_recognized_at_admission(tmp_path, monkeypatch):
    """A Pilatus calibration documents a counter limit, so the Run resolves an automatic saturation ceiling (decision
    39) and its outputs record that identity: admission signs the ceiling before recognizing completed scans, which
    are then skipped without a capture, materialization or warning (R10's case, with TIFF + .txt)."""
    from xrd_tools.sources.execution_graph import SourceFileState

    raw = _completed_and_new(tmp_path, shape=PILATUS)
    captured = []
    real_capture = SourceFileState.capture

    def capture(cls, path):
        captured.append(os.path.basename(str(path)))
        return real_capture(path)

    monkeypatch.setattr(SourceFileState, "capture", classmethod(capture))
    seams = _Seams(monkeypatch)
    events, _catalog = _run(tmp_path, raw, 9106, shape=PILATUS)
    warnings = [warning for event in events if event.kind in _TERMINAL for warning in event.warnings]
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"] and warnings == []
    assert seams.materialized == ["scanb_int1d.nexus"]
    assert not any(name.startswith("scana_") for name in captured), captured


def test_a_growing_scan_appends_in_place_without_copying_or_hashing_its_output(tmp_path, monkeypatch):
    """Maintainer, 2026-10-23, "Append in place": the processed frames are neither copied nor hashed again; only the
    new frame is written, into the same file."""
    import numpy as np
    import xrd_tools.io.output_transaction as output_transaction
    raw = tmp_path / "raw"
    raw.mkdir()
    _scan(raw, "scana", 3)
    events, _catalog = _run(tmp_path, raw, 9111)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    output = tmp_path / "processed" / "scana_int1d.nexus"
    with h5py.File(output, "r") as handle:
        processed = handle["entry/integrated_1d/intensity"][()]
    inode = output.stat().st_ino
    _scan(raw, "scana", 1, start=3)
    touched = []
    real_stage = output_transaction.OutputTransaction._stage_prior
    real_copy = output_transaction.OutputTransaction._copy_stream_seed
    real_hash = output_transaction._sha256_handle

    def stage(transaction, *args, **kwargs):
        touched.append(("stage", os.path.basename(transaction.admission.target)))
        return real_stage(transaction, *args, **kwargs)

    def copy(transaction, *args, **kwargs):
        touched.append(("copy", os.path.basename(transaction.admission.target)))
        return real_copy(transaction, *args, **kwargs)

    def hashed(handle):
        touched.append(("hash", os.fstat(handle.fileno()).st_ino))
        return real_hash(handle)

    monkeypatch.setattr(output_transaction.OutputTransaction, "_stage_prior", stage)
    monkeypatch.setattr(output_transaction.OutputTransaction, "_copy_stream_seed", copy)
    monkeypatch.setattr(output_transaction, "_sha256_handle", hashed)
    events, _catalog = _run(tmp_path, raw, 9112)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert touched == []  # the Run's only output transaction is scana's
    assert output.stat().st_ino == inode and _labels(tmp_path, "scana") == (1, 2, 3, 4)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["entry/integrated_1d/intensity"][:3], processed)
        assert handle["entry/integrated_1d/intensity"].shape[0] == 4


@pytest.mark.parametrize("first_new", (False, True), ids=("completed-then-completed", "new-then-completed"))
def test_an_already_processed_scan_after_another_scan_is_shown_from_its_saved_frames(tmp_path, monkeypatch, first_new):
    """Beamline 2026-10-23 (dev 5ebc1307, Windows share, 35 of 68 scans already processed): the second scan an Append
    found already processed failed the Run, "a sub-scan boundary needs a complete (scan key, source, display scan)
    selection". It is never read again, so it has no scan to show, only its saved frames."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _scan(raw, "scanb", 2)
    if not first_new:
        _scan(raw, "scana", 3)
    events, _catalog = _run(tmp_path, raw, 9201)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    if first_new:
        _scan(raw, "scana", 3)
    _scan(raw, "scanc", 1)
    seams = _Seams(monkeypatch)
    events, catalog = _run(tmp_path, raw, 9202)
    rows, warnings, navigation = _shown(events, catalog)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"], rows[-1]
    assert rows[-1][4] == "Complete · 6 processed · 0 skipped · 0 pending · 6 discovered" and warnings == []
    assert navigation == [("scana", 1, 1), ("scana", 2, 2), ("scana", 3, 3), ("scanb", 1, 4), ("scanb", 2, 5),
                          ("scanc", 1, 6)]
    assert seams.materialized == ["scana_int1d.nexus", "scanc_int1d.nexus"][(0 if first_new else 1):]
    assert _labels(tmp_path, "scana") == (1, 2, 3) and _labels(tmp_path, "scanb") == (1, 2)
    assert _labels(tmp_path, "scanc") == (1,)


def test_an_in_place_append_seals_its_close_after_its_writing_handle_closes(tmp_path, monkeypatch):
    """Beamline 2026-10-23 (dev d28d0716, Windows share): a growing scan's in-place Append failed, "held stream
    terminal (fast or in-place) lost its close checkpoint". Its close was sealed while the writer's duplicate of its
    writing handle was still open, and Windows finalizes a written file's timestamps when that handle closes; here
    that close moves mtime 86.5 ms, as the share did for the stream seed (48a7ee5d)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _scan(raw, "scana", 3)
    events, _catalog = _run(tmp_path, raw, 9301)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    _scan(raw, "scana", 1, start=3)
    output = tmp_path / "processed" / "scana_int1d.nexus"
    duplicates = set()

    class _ClosingFinalizesTimestamps:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def dup(descriptor):
            duplicate = os.dup(descriptor)
            duplicates.add(duplicate)
            return duplicate

        @staticmethod
        def close(descriptor):
            os.close(descriptor)
            if descriptor in duplicates:
                duplicates.discard(descriptor)
                state = output.stat()
                os.utime(output, ns=(state.st_atime_ns, state.st_mtime_ns + 86_500_000))

    monkeypatch.setattr(record_writer, "os", _ClosingFinalizesTimestamps())
    events, _catalog = _run(tmp_path, raw, 9302)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"], events[-1].detail
    assert duplicates == set() and _labels(tmp_path, "scana") == (1, 2, 3, 4)
