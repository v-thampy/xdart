"""A Folder Append's admission asks the file system little about raw files it already holds (NEXT-1215-SHARE-LOOKUPS;
maintainer, 2026-11-19, of the 1.2.14 beamline run: "The admission step still takes about 14 s on your share.").

Admission used to stat every listed file, sidecars included, and resolve and stat every raw image for the collision
guard. The real page runs Folder Append over saved scans. The admission thread's Python calls on the listed files
are counted through the real os functions. This does not count filesystem work internal to those functions."""
from dataclasses import replace
import logging
import os
import threading

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _terminal
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans, _written
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


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
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
    assert terminal.files_processed == 120 and terminal.frames_new == 0
    # The debug log splits admission's time, so a slow share shows which part to fix (maintainer, 2026-11-19:
    # "the initial start is still pretty slow"; his pick "Add the timings first (Recommended)").
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("[DIRECTORY-LIST] 120 files listed (") for line in lines), lines
    admit = [line for line in lines if line.startswith("[APPEND-ADMIT] ")]
    assert len(admit) == 1 and "already-processed=30" in admit[0], admit
    assert all(part in admit[0] for part in ("saved records", "raw-file guard", "scans to process")), admit
    total = sum(calls.values())
    # Ordinary POSIX inputs need one stat. Windows retains one canonical-path
    # check, which does not call Python's os.lstat on this fixture.
    assert 0 < total <= 2 * images, f"admission made {calls} raw calls for {images} images"
