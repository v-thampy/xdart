"""A later Append in the same session does not open the saved outputs it already read (NEXT-1215-SHARE-LOOKUPS).

Admission recognizes an already-processed scan from its output's committed record: an HDF5 open per saved output,
about 22 ms each on the beamline share (maintainer, 2026-11-19, 1.2.15: "saved records 7104 ms" for 327 outputs; his
pick "Remember saved records (Recommended)"). The record is remembered by the output's file revision; replacement
and in-place changes require another read. The real page runs the Folder Appends; HDF5 opens are counted at
h5py's file-open seam on the admission thread."""
from dataclasses import replace
import os
import shutil
import sys
import threading

import pytest

pytest.importorskip("pyqtgraph")

import h5py._hl.files as h5_files

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _terminal
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans, _written
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


def test_a_second_append_reads_only_the_saved_outputs_that_changed(qapp, tmp_path, monkeypatch):  # noqa: F811
    stems, intent = _scans(tmp_path, 30, 2)
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 60
    assert _written(tmp_path, stems)
    processed = os.path.normcase(os.fspath(tmp_path / "processed")) + os.sep
    opened: list[str] = []
    real_open = h5_files.make_fid

    def counted(name, *args, **kwargs):
        if threading.current_thread().name == "scattering-admission" and isinstance(name, (str, bytes)):
            spelling = os.path.normcase(os.fsdecode(name))
            if spelling.startswith(processed):
                opened.append(os.path.basename(spelling))
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(h5_files, "make_fid", counted)

    def append():
        opened.clear()
        with _page(qapp, replace(intent, output_mode="Append")) as (page, _view, observed):
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
        assert terminal.files_processed == 60 and terminal.frames_new == 0
        return sorted(opened)

    assert len(append()) == 30                     # the first Append reads every saved record
    assert append() == []                          # the next one remembers them all
    touched = tmp_path / "processed" / f"{stems[3]}_int1d.nexus"
    stat = touched.stat()
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert append() == [os.path.normcase(touched.name)]  # a changed output is read again
    replacement = touched.with_suffix(".replace")
    shutil.copy2(touched, replacement)
    stat = touched.stat()
    replacement.replace(touched)
    assert (touched.stat().st_size, touched.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    assert append() == [os.path.normcase(touched.name)]  # same size and time, different file
    with touched.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(byte)
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert append() == [os.path.normcase(touched.name)]  # same inode, size and mtime, later change time


def test_a_second_live_append_reads_only_the_saved_outputs_that_changed(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Live Append uses the same remembered records (maintainer, 2026-11-19: "these fixes should apply as generally as
    possible"; his pick "Hold 1.2.16 for Live"). Each Live poll already reads a saved output only when its scan or its
    output changed; a second Live Run in the session read every one again."""
    from tests.xdart.scattering.test_browse_reference_rings_run import _pump, _TERMINAL
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    stems, intent = _scans(tmp_path, 30, 2)
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 60
    assert _written(tmp_path, stems)
    processed = os.path.normcase(os.fspath(tmp_path / "processed")) + os.sep
    opened: list[str] = []
    fingerprints: list[int] = []
    fingerprint_counts: list[int] = []
    real_open = h5_files.make_fid
    science_projection = dynamic_output._science_projection

    def counted_projection(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_append_science_fingerprint":
            fingerprints.append(1)
        return science_projection(*args, **kwargs)

    def reading_a_record() -> bool:  # Live also opens its newest scan's output for its own epoch; count records only
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code.co_name == "_committed_record":
                return True
            frame = frame.f_back
        return False

    def counted(name, *args, **kwargs):
        if isinstance(name, (str, bytes)) and reading_a_record():
            spelling = os.path.normcase(os.fsdecode(name))
            if spelling.startswith(processed):
                opened.append(os.path.basename(spelling))
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(h5_files, "make_fid", counted)
    monkeypatch.setattr(dynamic_output, "_science_projection", counted_projection)

    def live():
        opened.clear()
        fingerprints.clear()
        with _page(qapp, replace(intent, output_mode="Append", live_mode=True)) as (page, _view, observed):
            page._shell.run_controls.startButton.click()
            assert _pump(qapp, lambda: any((getattr(e, "files_processed", 0) or 0) >= 60 for e in observed),
                         seconds=120), "the saved scans were never all counted"
            counted_opens = sorted(opened)
            fingerprint_counts.append(len(fingerprints))
            page._shell.run_controls.stopButton.click()
            assert _pump(qapp, lambda: any(e.kind in _TERMINAL for e in observed), seconds=120)
        return counted_opens

    assert len(set(live())) == 30                  # the first Live Run reads every saved record
    assert live() == []                            # the next one remembers them all
    touched = tmp_path / "processed" / f"{stems[5]}_int1d.nexus"
    stat = touched.stat()
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert set(live()) == {os.path.normcase(touched.name)}  # a changed output is read again
    assert fingerprint_counts == [1, 1, 1]
