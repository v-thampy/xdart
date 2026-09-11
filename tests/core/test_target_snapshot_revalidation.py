"""Content-bound Browse snapshots can reuse an unchanged file revision."""

from dataclasses import replace
import os

import pytest

from xrd_tools.io import output_transaction as module


@pytest.mark.parametrize("change", ("same-stat-write", "replace", "delete"))
def test_revalidation_refuses_changed_original(tmp_path, change):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    before = path.stat()
    snapshot = module.capture_target_snapshot(path)
    if change == "same-stat-write":
        path.write_bytes(b"modified")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_size == before.st_size
        assert path.stat().st_mtime_ns == before.st_mtime_ns
    elif change == "replace":
        other = tmp_path / "other.nexus"
        other.write_bytes(b"original")
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    else:
        path.unlink()
    with pytest.raises(module.TargetChanged):
        module.revalidate_target_snapshot(path, snapshot)


def test_revalidation_preserves_digest_and_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    snapshot = module.capture_target_snapshot(path)
    descriptors = []
    real_open = module.os.open

    def record_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(module.os, "open", record_open)
    assert module.revalidate_target_snapshot(path, snapshot) is snapshot
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert module.capture_target_snapshot(path) == snapshot


def test_snapshot_without_revision_keeps_full_content_check(tmp_path):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    snapshot = module.capture_target_snapshot(path)
    legacy = replace(snapshot, ctime_ns=None)
    assert module.revalidate_target_snapshot(path, legacy) == snapshot
    before = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(module.TargetChanged):
        module.revalidate_target_snapshot(path, legacy)
