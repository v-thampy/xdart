"""A Folder Append's admission asks the file system little about raw files it already holds (NEXT-1215-SHARE-LOOKUPS;
maintainer, 2026-11-19, of the 1.2.14 beamline run: "The admission step still takes about 14 s on your share.").

On a network share every stat or resolve is a round trip. Admission used to stat every listed file, sidecars included,
and resolve and stat every raw image for the output-collision guard: about six calls per image. The real page runs a
Folder Append over saved scans; the admission thread's calls on raw paths are counted through the real os functions."""
from dataclasses import replace
import os
import threading

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _terminal
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans, _written
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


@pytest.mark.parametrize("recursive", (False, True), ids=("flat", "recursive"))
def test_folder_append_admission_makes_about_one_raw_lookup_per_image(qapp, tmp_path, monkeypatch, recursive):  # noqa: F811
    stems, intent = _scans(tmp_path, 30, 4)
    intent = replace(intent, source_spec=replace(intent.source_spec, recursive=recursive))
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 120
    assert _written(tmp_path, stems)

    raw = tmp_path / "raw"
    images = len(list(raw.glob("*.tif")))
    assert images == 120 and len(list(raw.iterdir())) == 240  # each image has its sidecar
    raw_root = os.path.normcase(os.fspath(raw))
    calls: dict[str, int] = {}

    def counted(name, real):
        def call(path, *args, **kwargs):
            if threading.current_thread().name == "scattering-admission" and not isinstance(path, int):
                spelling = os.path.normcase(os.fsdecode(os.fspath(path)))
                if spelling == raw_root or spelling.startswith(raw_root + os.sep):
                    calls[name] = calls.get(name, 0) + 1
            return real(path, *args, **kwargs)
        return call

    with _page(qapp, replace(intent, output_mode="Append")) as (page, _view, observed):
        with monkeypatch.context() as patch:
            for module, name in ((os, "stat"), (os, "lstat"), (os.path, "realpath")):
                patch.setattr(module, name, counted(name, getattr(module, name)))
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
    assert terminal.files_processed == 120 and terminal.frames_new == 0
    total = sum(calls.values())
    # One stat per image remains: the collision guard's identity check. 1.2.14 made about six per image.
    assert 0 < total <= 2 * images, f"admission made {calls} raw calls for {images} images"
