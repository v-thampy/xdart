"""A Folder Append's admission asks the file system little about raw files it already holds (NEXT-1215-SHARE-LOOKUPS;
maintainer, 2026-11-19, of the 1.2.14 beamline run: "The admission step still takes about 14 s on your share.").

Admission used to stat every listed file, sidecars included, and resolve and stat every raw image for the collision
guard. The real page runs Folder Append over saved scans. The admission thread's Python calls on the listed files
are counted through the real os functions. This does not count filesystem work internal to those functions."""
from dataclasses import replace
import logging
import os
import sys
import threading

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _terminal
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans, _written
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import dynamic_output


@pytest.mark.parametrize("recursive", (False, True), ids=("flat", "recursive"))
def test_folder_append_admission_bounds_raw_file_lookups(qapp, tmp_path, monkeypatch, caplog, recursive):  # noqa: F811
    stems, intent = _scans(tmp_path, 30, 4)
    intent = replace(intent, source_spec=replace(intent.source_spec, recursive=recursive))
    raw = tmp_path / "raw"
    if recursive:
        subfolder = raw / "subfolder"
        subfolder.mkdir()
        moved_stems = set(stems[::2])
        for path in tuple(raw.iterdir()):
            if path.stem.rsplit("_", 1)[0] in moved_stems:
                path.rename(subfolder / path.name)
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 120
    assert _written(tmp_path, stems)

    images = len(list(raw.rglob("*.tif")))
    listed_files = {os.path.normcase(os.fspath(path)) for path in raw.rglob("*") if path.is_file()}
    assert images == 120 and len(listed_files) == 240  # each image has its sidecar
    calls: dict[str, int] = {}
    lexical_calls = science_calls = 0
    abspath = os.path.abspath
    science_projection = dynamic_output._science_projection

    def counted_abspath(path):
        nonlocal lexical_calls
        if (sys._getframe(1).f_globals.get("__name__") == output_preflight.__name__
                and os.path.normcase(os.fsdecode(path)) in listed_files):
            lexical_calls += 1
        return abspath(path)

    def counted_science(*args, **kwargs):
        nonlocal science_calls
        if threading.current_thread().name == "scattering-admission":
            science_calls += 1
        return science_projection(*args, **kwargs)

    def counted(name, real):
        def call(path, *args, **kwargs):
            if threading.current_thread().name == "scattering-admission" and not isinstance(path, int):
                spelling = os.path.normcase(os.fsdecode(os.fspath(path)))
                if spelling in listed_files:
                    calls[name] = calls.get(name, 0) + 1
            return real(path, *args, **kwargs)
        return call

    with _page(qapp, replace(intent, output_mode="Append")) as (page, _view, observed):
        caplog.set_level(logging.DEBUG, logger="xdart.gui.tabs.scattering.output_preflight")
        with monkeypatch.context() as patch:
            for module, name in ((os, "stat"), (os, "lstat"), (os.path, "realpath")):
                patch.setattr(module, name, counted(name, getattr(module, name)))
            patch.setattr(os.path, "abspath", counted_abspath)
            patch.setattr(dynamic_output, "_science_projection", counted_science)
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
    assert terminal.files_processed == 120 and terminal.frames_new == 0
    # The debug log splits admission's time, so a slow share shows which part to fix (maintainer, 2026-11-19:
    # "the initial start is still pretty slow"; his pick "Add the timings first (Recommended)").
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("[DIRECTORY-LIST] 120 files listed (") for line in lines), lines
    admit = [line for line in lines if line.startswith("[APPEND-ADMIT] ")]
    assert len(admit) == 1 and "already-processed=30" in admit[0], admit
    assert all(part in admit[0] for part in ("saved records", "raw-file guard", "the rest (grouping and scans to process)")), admit
    total = sum(calls.values())
    # Ordinary POSIX inputs need one stat. Windows retains one canonical-path
    # check, which does not call Python's os.lstat on this fixture.
    assert 0 < total <= 2 * images, f"admission made {calls} raw calls for {images} images"
    assert (0 < lexical_calls <= 4 * images, science_calls) == (True, 1), (lexical_calls, science_calls)


@pytest.mark.parametrize("recursive", (False, True), ids=("flat", "recursive"))
def test_the_windows_guard_resolves_raw_files_folder_by_folder(qapp, tmp_path, monkeypatch, recursive):  # noqa: F811
    """On Windows the raw-file guard still compares canonical paths (Codex review 9f0471bc), and used to resolve every
    raw image: 7.0 s for 930 on the maintainer's share (his pick "Cheaper raw-file guard"). It now resolves each folder
    once and builds a plain file's canonical path from its folder's and its listed name."""
    stems, intent = _scans(tmp_path, 30, 4)
    intent = replace(intent, source_spec=replace(intent.source_spec, recursive=recursive))
    if recursive:  # his Folder is recursive: half the scans in a subfolder
        raw = tmp_path / "raw"
        (raw / "subfolder").mkdir()
        for path in tuple(raw.iterdir()):
            if path.is_file() and path.stem.rsplit("_", 1)[0] in set(stems[::2]):
                path.rename(raw / "subfolder" / path.name)
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 120
    raw_root = os.path.normcase(os.fspath(tmp_path / "raw"))
    resolved: list[str] = []
    realpath = os.path.realpath

    def counted(path, *args, **kwargs):
        spelling = os.path.normcase(os.fsdecode(os.fspath(path)))
        if threading.current_thread().name == "scattering-admission" and (
                spelling == raw_root or spelling.startswith(raw_root + os.sep)):
            resolved.append(spelling)
        return realpath(path, *args, **kwargs)

    monkeypatch.setattr(output_preflight, "_CHECK_RESOLVED_RAW_PATHS", True)
    with _page(qapp, replace(intent, output_mode="Append")) as (page, _view, observed):
        with monkeypatch.context() as patch:
            patch.setattr(os.path, "realpath", counted)
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
    assert terminal.files_processed == 120 and terminal.frames_new == 0
    # 1.2.15's Windows branch resolved each of the 120 images; now a few per Run: each folder once, and the source
    # root for the folder policy (5 here for the recursive Folder's two folders).
    assert raw_root in resolved
    if recursive:
        assert os.path.join(raw_root, "subfolder") in resolved
    assert len(resolved) <= 8, resolved
