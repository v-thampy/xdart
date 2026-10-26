"""A Folder Run's scans leave its Save Path out (beamline 2026-10-25).

Live, Subdirs on, Save Path ``xdart_processed_data`` inside the source folder: "Save Path ... is inside the
recursively-watched source directory ...; the processed output would be re-discovered as a raw input". The
maintainer: "there is no ambiguity in the save path as it is explicit, and it shouldn't be included in the
processing when using Run"; his pick "Skip the Save Path": every scan of the folder leaves it out, the refusal goes
for every File Type, and a Save Path that is the source folder itself stays refused for HDF5/NeXus folders only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xrd_tools.io.output_safety import OutputCollisionError, check_output_not_source, scan_skip_dirs
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.discover import collect_owned_candidates
from xrd_tools.sources.selection import DirectorySourceSpec


def _folder(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "Baniecki_Oct2026"
    (raw / "calibration").mkdir(parents=True)
    for index in (1, 2):
        _frame(raw / "calibration", "lab6", index, "txt", float(index))
    _frame(raw, "top", 1, "txt", 1.0)
    save = raw / "xdart_processed_data"
    (save / "calibration").mkdir(parents=True)
    _frame(save, "stray", 1, "txt", 9.0)  # an image inside the Save Path is never Run input
    _frame(save / "calibration", "stray", 2, "txt", 9.0)
    return raw, save


def test_scan_skip_dirs_names_a_save_path_strictly_inside_the_folder(tmp_path) -> None:
    raw, save = _folder(tmp_path)
    assert scan_skip_dirs(raw, save) == (str(save),)
    assert scan_skip_dirs(raw, raw) == ()                           # the folder itself cannot be skipped
    assert scan_skip_dirs(raw, tmp_path / "elsewhere") == ()
    assert scan_skip_dirs(raw, save / "scan_int2d.nexus") == ()     # a file Save Path names one output


def test_every_scan_of_the_folder_leaves_the_save_path_out(tmp_path) -> None:
    raw, save = _folder(tmp_path)
    skip = scan_skip_dirs(raw, save)
    listed = {path.relative_to(raw).as_posix() for path in (
        candidate.path for candidate in collect_owned_candidates(raw, recursive=True, skip=skip))}
    assert {"top_0001.tif", "calibration/lab6_0001.tif", "calibration/lab6_0002.tif"} <= listed, listed
    assert not any(name.startswith("xdart_processed_data/") for name in listed), listed
    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(raw, recursive=True, suffixes=(".tif",), skip=skip)
        names = sorted(c.path.name for c in session.observe(refresh=True).discovered_snapshot.candidates)
        assert names == ["lab6_0001.tif", "lab6_0002.tif", "top_0001.tif"]
        session.configure(raw, recursive=True, suffixes=(".tif",))
        assert len(session.observe(refresh=True).discovered_snapshot.candidates) == 5  # without the skip
    finally:
        session.close()


def test_the_output_check_refuses_only_what_a_scan_could_read_back(tmp_path) -> None:
    raw, save = _folder(tmp_path)
    inside = save / "calibration" / "lab6_int2d.nexus"
    check_output_not_source(inside, watched_dirs=(raw,), recursive=True, container_directory_mode=True,
                            skipped_dirs=scan_skip_dirs(raw, save))
    with pytest.raises(OutputCollisionError, match="recursively-watched"):
        check_output_not_source(inside, watched_dirs=(raw,), recursive=True, container_directory_mode=True)
    with pytest.raises(OutputCollisionError, match="raw container source directory"):
        check_output_not_source(raw / "scan_int2d.nexus", watched_dirs=(raw,), recursive=True,
                                container_directory_mode=True, skipped_dirs=scan_skip_dirs(raw, raw))


def test_a_tif_folder_run_with_its_save_path_inside_runs_and_reads_nothing_there(tmp_path) -> None:
    raw, save = _folder(tmp_path)
    intent = replace(
        _intent(tmp_path, raw, live=False),
        source_spec=DirectorySourceSpec(raw, recursive=True, suffixes=(".tif",), metadata_format="txt"),
        save_path=str(save),
    )
    executor, identity, events = _run_to_terminal(intent, request_value=9951)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert (terminals[0].files_discovered, terminals[0].files_processed) == (3, 3)
    written = sorted(path.relative_to(save).as_posix() for path in save.rglob("*.nexus"))
    assert written and not any("stray" in name for name in written), written
