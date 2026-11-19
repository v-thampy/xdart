"""Directory Append must bound raw-path resolution when it finds saved scans."""
from dataclasses import replace
import os

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _terminal
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans, _written
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


def test_folder_append_resolves_raw_paths_at_most_linearly(qapp, tmp_path, monkeypatch):  # noqa: F811
    stems, intent = _scans(tmp_path, 30, 4)
    with _page(qapp, intent) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _terminal(qapp, page, observed).completed == 120
    assert _written(tmp_path, stems)

    raw = tmp_path / "raw"
    listed_files = len(list(raw.iterdir()))
    assert listed_files == 240  # 120 TIFFs and their 120 sidecars
    raw_root = os.path.normcase(os.fspath(raw))
    raw_prefix = raw_root + os.sep
    calls = 0
    realpath = os.path.realpath

    def counted_realpath(path, *args, **kwargs):
        nonlocal calls
        spelling = os.path.normcase(os.fsdecode(path))
        if spelling == raw_root or spelling.startswith(raw_prefix):
            calls += 1
        return realpath(path, *args, **kwargs)

    with _page(qapp, replace(intent, output_mode="Append")) as (page, _view, observed):
        with monkeypatch.context() as patch:
            patch.setattr(os.path, "realpath", counted_realpath)
            page._shell.run_controls.startButton.click()
            terminal = _terminal(qapp, page, observed)
    assert terminal.files_processed == 120 and terminal.frames_new == 0
    # Allow more than the measured three probes per listed file, independent of disk latency.
    assert 0 < calls <= 4 * listed_files, f"{calls} raw realpath calls for {listed_files} listed files"
