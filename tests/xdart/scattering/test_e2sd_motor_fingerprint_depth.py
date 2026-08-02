from __future__ import annotations

import os
from pathlib import Path

import h5py

from tests.xdart.scattering._e2sd_support import write_motor_container
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xrd_tools.sources.selection import DirectorySourceSpec


def test_same_size_mtime_motor_rewrite_invalidates_cached_knowledge(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    candidate = raw / "scan.nxs"
    write_motor_container(candidate, "halpha")
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    request = SourceObservationRequest(1, 0, source)
    adapter = FilesystemSourceAdapter()

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)
    assert preview.gi_motor_choices == ("halpha",)
    adapter.publish_motor_knowledge(preview)

    before = candidate.stat()
    with h5py.File(candidate, "r+") as handle:
        handle.move(
            "entry/instrument/positioners/halpha",
            "entry/instrument/positioners/zzzzzz",
        )
    os.utime(candidate, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = candidate.stat()
    assert (after.st_size, after.st_mtime_ns) == (
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns

    current = adapter.observe(SourceObservationRequest(2, 1, source))
    assert current.candidate_fingerprint != passive.candidate_fingerprint
    assert adapter.project_motor_knowledge(
        source, current.candidate_fingerprint
    ) is None


def test_passive_source_fingerprinting_never_opens_candidate_content(
    tmp_path: Path, monkeypatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    candidate = raw / "scan.nxs"
    write_motor_container(candidate, "halpha")
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))

    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("passive fingerprint opened candidate content")
        ),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("passive fingerprint opened candidate content")
        ),
    )

    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, source)
    )
    assert observed.exists
    assert observed.direct_child_count == 1
    assert observed.candidate_fingerprint
