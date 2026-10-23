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
from tests.xdart.scattering.test_live_revalidation_cost_run import TINY, _poni, _Sidecar, _tiff
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

_ENERGY = "Using the scan energy, 16.000 keV; the PONI says 12.398 keV."


def _intent(tmp_path, raw, *, mode="Append", npt=8, processing="Int 1D"):
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
        poni_file=str(_poni(tmp_path, TINY)), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode=mode, processing_mode=processing, live_mode=False, max_cores=1,
        bai_1d_args={"npt": npt, "method": "numpy"}, bai_2d_args={"npt_rad": 8, "npt_azim": 8, "method": "numpy"})


def _scan(raw, stem, frames, start=0):
    scan = _Sidecar(raw, TINY, stem)
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


def _completed_and_new(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _scan(raw, "scana", 3)
    events, _catalog = _run(tmp_path, raw, 9101)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    _scan(raw, "scanb", 2)
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
    if change in ("renamed", "raw-rewritten", "settings"):
        warnings = [warning for event in events if event.kind in _TERMINAL for warning in event.warnings]
        assert any("scana" in warning for warning in warnings), warnings
