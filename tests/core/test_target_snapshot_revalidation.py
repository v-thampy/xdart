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


# PERF-BROWSE-OPEN-128 stage A (Codex design review 2, A1-A3): read-only Browse takes a revision seal (the file's
# identity and descriptor ctime, no content digest); revalidation reuses the stat+ctime fence it already trusted.

def _no_hash(monkeypatch):
    def forbidden(_handle):
        raise AssertionError("a revision seal must not hash file content")
    monkeypatch.setattr(module, "_sha256_handle", forbidden)


def test_revision_capture_reads_no_content_and_records_the_descriptor_revision(tmp_path, monkeypatch):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    _no_hash(monkeypatch)
    revision = module.capture_target_revision(path)
    observed = os.stat(path)
    assert revision.exists and revision.digest is None and revision.ctime_ns is not None
    assert (revision.size, revision.mtime_ns, revision.device, revision.inode) == (
        observed.st_size, observed.st_mtime_ns, observed.st_dev, observed.st_ino)
    assert module.revalidate_target_snapshot(path, revision) is revision, "an unchanged revision is reused, unhashed"


@pytest.mark.parametrize("change", ("append", "same-stat-write", "replace", "delete"))
def test_revision_revalidation_refuses_a_changed_file_without_hashing(tmp_path, monkeypatch, change):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    before = path.stat()
    revision = module.capture_target_revision(path)
    _no_hash(monkeypatch)
    if change == "append":
        with open(path, "ab") as handle:
            handle.write(b"x")
    elif change == "same-stat-write":
        path.write_bytes(b"modified")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "replace":
        other = tmp_path / "other.nexus"
        other.write_bytes(b"original")
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    else:
        path.unlink()
    with pytest.raises(module.TargetChanged):
        module.revalidate_target_snapshot(path, revision)


def test_revision_capture_of_a_missing_file_is_absent(tmp_path):
    assert not module.capture_target_revision(tmp_path / "missing.nexus").exists


def test_revision_key_tells_a_ctime_only_change_apart_where_snapshot_equality_cannot(tmp_path):
    path = tmp_path / "scan.nexus"
    path.write_bytes(b"original")
    before = path.stat()
    first = module.capture_target_revision(path)
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = module.capture_target_revision(path)
    if second.ctime_ns == first.ctime_ns:
        pytest.skip("this filesystem did not advance ctime")
    assert first == second, "TargetSnapshot equality ignores ctime (compare=False) and a revision has no digest"
    assert module.target_revision_key(first) != module.target_revision_key(second)
