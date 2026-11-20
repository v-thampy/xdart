"""A later Append in the same session does not open the saved outputs it already read (NEXT-1215-SHARE-LOOKUPS).

Admission recognizes an already-processed scan from its output's committed record: an HDF5 open per saved output,
about 22 ms each on the beamline share (maintainer, 2026-11-19, 1.2.15: "saved records 7104 ms" for 327 outputs; his
pick "Remember saved records (Recommended)"). The record is remembered by the output's file revision; replacement
and in-place changes require another read. The real page runs the Folder Appends; HDF5 opens are counted at
h5py's file-open seam on the admission thread."""
from dataclasses import replace
import os
import shutil
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
