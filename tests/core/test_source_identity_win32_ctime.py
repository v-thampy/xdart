"""Live and Append compare an already-processed file with the win32 ctime seam (beamline 2026-10-25).

On the maintainer's Windows share only ``ctime_ns`` (a pathname stat's creation time there) moved after a frame
was published, +3 ms to +325 ms, with size, mtime, device and inode unchanged; every Live output then stopped
growing. A real ``chmod`` moves this platform's ctime the same way and nothing else. On Windows ``chmod`` cannot
move the creation time, so these run on POSIX; his beamline re-test is the native check.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceSpec
from xrd_tools.sources import execution_graph as graph
from xrd_tools.sources.selection import image_series_spec

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="chmod moves POSIX ctime only; Windows reports the creation time")

_SIDECAR = "# Counters\nI0 = 3.0\n# Motors\n\nUser: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"


def _series(tmp_path: Path) -> SourceSpec:
    for index in (1, 2):
        tifffile.imwrite(tmp_path / f"scan_{index:04d}.tif", np.full((2, 3), index, dtype=np.uint16))
        (tmp_path / f"scan_{index:04d}.txt").write_text(_SIDECAR)
    source = image_series_spec(tmp_path / "scan_0001.tif", metadata_format="txt")
    return SourceSpec(source.uri, source.kind, options={**dict(source.options), "metadata_format": "txt"})


def _move_ctime(path: Path) -> None:
    before = os.stat(path)
    time.sleep(0.02)
    os.chmod(path, before.st_mode)
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) == (
        before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
    assert after.st_ctime_ns != before.st_ctime_ns


def _published_then_settled(tmp_path: Path):
    source = _series(tmp_path)
    published = graph.qualify_source_execution_graph(source)
    assert [item.metadata_file is not None for item in published.stamp.metadata_sources] == [True, True]
    proof = graph.LivePrefixProof(published, metadata_format=None)
    for name in ("scan_0001.tif", "scan_0001.txt", "scan_0002.txt"):
        _move_ctime(tmp_path / name)
    return source, published, proof, graph.qualify_source_execution_graph(source)


def test_win32_identity_keeps_a_published_frame_whose_ctime_settled(tmp_path, monkeypatch, ctime_seam):
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
    _source, published, proof, again = _published_then_settled(tmp_path)

    assert proof.holds()
    assert (graph.append_source_from_execution_graph(again, generation=0).digest
            == graph.append_source_from_execution_graph(published, generation=0).digest)
    # The capture stays exact (its other consumers compare it with raw stats): only the comparisons settle ctime.
    assert again.stamp.members != published.stamp.members


def test_posix_identity_still_sees_a_ctime_only_change(tmp_path, monkeypatch, ctime_seam):
    monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", True)
    _source, published, proof, again = _published_then_settled(tmp_path)

    assert not proof.holds()
    assert (graph.append_source_from_execution_graph(again, generation=0).digest
            != graph.append_source_from_execution_graph(published, generation=0).digest)
    # The members moved too, so the drift fence leaves this to the raw-source validators, as before.
    assert graph.published_metadata_drift(published, again) == ()
    assert again.stamp.members != published.stamp.members


def test_drift_fence_and_a_sidecar_whose_ctime_alone_moved(tmp_path, monkeypatch, ctime_seam):
    """The first beamline warning: "the metadata of frames it already processed changed (..._0003.txt, ...)"."""
    for carries, folder in ((False, tmp_path / "win32"), (True, tmp_path / "posix")):
        folder.mkdir()
        monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", carries)
        source = _series(folder)
        published = graph.qualify_source_execution_graph(source)
        _move_ctime(folder / "scan_0002.txt")
        again = graph.qualify_source_execution_graph(source)

        assert graph.published_metadata_drift(published, again) == (
            (str(folder / "scan_0002.txt"),) if carries else ())
