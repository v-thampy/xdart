"""B-1 of NEXT-1210-VERIFY-ONCE: a Stitch or RSM save verifies its result once and reuses that evidence.

One whole-file hash of the closed candidate and one semantic inspection of it; file-revision fences (no rehash)
carry both through the owned rename; the old target of a Replace is held by its revision key, never rehashed. The
counters are a route oracle: exact, so a hash moved to another stream of this path reads 0 and fails. The receipt's
digest and inspection are then checked independently, outside the counted window.

A retry after a published rename whose post-rename check failed: with that check's revision recorded, an exact
revision compare; without it, one hash against the candidate's digest (the maintainer's "Hash once on that retry
(Recommended)", 2026-11-12). Neither inspects again.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import xrd_tools.io.analysis_artifact as artifact_api
import xrd_tools.io.finite_artifact as finite_api
import xrd_tools.io.output_transaction as transaction_api
from tests.core.test_analysis_artifact import _request, _writer
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    admit_analysis_artifact,
    inspect_analysis_artifact,
)
from xrd_tools.io.output_transaction import OutputTransactionCoordinator

_KINDS = (AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.RSM)


class _Counts:
    """Whole-file hashes (by file name) and semantic inspections while ``counting`` is on."""

    def __init__(self, monkeypatch):
        self.counting = False
        self.hashed: list[str] = []
        self.finite_hashed: list[str] = []
        self.inspected: list[str] = []
        real_hash = transaction_api._sha256_handle
        real_capture = finite_api._capture_regular
        real_inspect = artifact_api.inspect_analysis_artifact

        def sha256_handle(handle):
            if self.counting:
                self.hashed.append(Path(handle.name).name)
            return real_hash(handle)

        def capture_regular(path, *args, **kwargs):
            # The finite owner's own hash stream. Its one use on this route hashes the empty reserved candidate
            # (0 bytes, before any write), so only a pass over content counts.
            if self.counting and kwargs.get("hash_content") and os.path.getsize(path) > 0:
                self.finite_hashed.append(Path(path).name)
            return real_capture(path, *args, **kwargs)

        def inspect(path, **kwargs):
            if self.counting:
                self.inspected.append(Path(path).name)
            return real_inspect(path, **kwargs)

        monkeypatch.setattr(transaction_api, "_sha256_handle", sha256_handle)
        monkeypatch.setattr(finite_api, "_capture_regular", capture_regular)
        monkeypatch.setattr(artifact_api, "inspect_analysis_artifact", inspect)

    def reset(self):
        self.hashed.clear()
        self.finite_hashed.clear()
        self.inspected.clear()


def _independent(receipt, request):
    """The receipt against a fresh, uncounted reading of the public file."""
    target = Path(request.target)
    assert receipt.terminal.digest == hashlib.sha256(target.read_bytes()).hexdigest()
    state = os.stat(target)
    assert (receipt.terminal.device, receipt.terminal.inode, receipt.terminal.size, receipt.terminal.mtime_ns) == (
        state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
    assert receipt.inspection == inspect_analysis_artifact(target, expected_request=request)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("mode", ("new", "replace"))
def test_a_save_hashes_its_result_once_and_inspects_it_once(tmp_path, monkeypatch, kind, mode):
    target = tmp_path / f"{mode}-{kind.value}.nexus"
    counts = _Counts(monkeypatch)
    overwrite, previous = AnalysisArtifactOverwrite.CREATE_NEW, None
    if mode == "replace":
        initial = _request(target, kind)
        previous = admit_analysis_artifact(initial, coordinator=OutputTransactionCoordinator()).publish(
            _writer(initial))
        overwrite = AnalysisArtifactOverwrite.REPLACE
    request = _request(target, kind, overwrite=overwrite)
    counts.counting = True
    receipt = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator()).publish(_writer(request))
    counts.counting = False
    # The parent: 7 hashes (4 of the candidate, 3 of the public file), 4 more of the old target on Replace, and a
    # second full inspection of the public file.
    assert len(counts.hashed) == 1 and counts.hashed[0].startswith(".xdart-finite-"), counts.hashed
    assert counts.finite_hashed == [], counts.finite_hashed
    assert len(counts.inspected) == 1 and counts.inspected[0].startswith(".xdart-finite-"), counts.inspected
    if previous is not None:
        assert receipt.terminal.inode != previous.terminal.inode
    _independent(receipt, request)


def _fail_first_post_rename_check(monkeypatch, request, name):
    """Fail the first call of ``name`` (module-level in analysis_artifact) on the public file, once:
    ``_post_rename_revision`` is the check right after the owned rename."""
    # The parent has no post-rename revision check; its stand-in there is never called.
    real = getattr(artifact_api, name, None) or transaction_api.capture_target_revision
    failed = []

    def fail_once(path, *args, **kwargs):
        if Path(path) == Path(request.target) and not failed:
            failed.append(name)
            raise OSError(f"{name} transient after the rename")
        return real(path, *args, **kwargs)

    # raising=False: on the parent, publish succeeds and the row fails at its first assertion.
    monkeypatch.setattr(artifact_api, name, fail_once, raising=False)
    return failed


def _edit_in_place_keeping_size_and_mtime(path: Path) -> None:
    """Change one byte, same inode, same size, mtime put back: only the content (and ctime) tell."""
    state = os.stat(path)
    with open(path, "r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF]))
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (state.st_ino, state.st_size, state.st_mtime_ns)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("edited", (False, True))
def test_a_retry_with_no_recorded_post_rename_revision_hashes_once(tmp_path, monkeypatch, kind, edited):
    request = _request(tmp_path / f"retry-unrecorded-{edited}.nexus", kind)
    output = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator())
    failed = _fail_first_post_rename_check(monkeypatch, request, "_post_rename_revision")
    writes = []

    def write_once(entry):
        writes.append("writer")
        _writer(request)(entry)

    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(write_once)
    assert failed and output.snapshot.published and output.snapshot.receipt is None
    assert not output.snapshot.slot_held
    edited_bytes = None
    if edited:
        _edit_in_place_keeping_size_and_mtime(Path(request.target))
        edited_bytes = Path(request.target).read_bytes()
    counts = _Counts(monkeypatch)
    counts.counting = True
    if edited:
        with pytest.raises(AnalysisArtifactCleanupPending) as pending:
            output.retry_cleanup()
        assert pending.value.snapshot.receipt is None and not pending.value.snapshot.slot_held
        assert Path(request.target).read_bytes() == edited_bytes  # refused, never rolled back
    else:
        recovered = output.retry_cleanup()
        counts.counting = False
        assert recovered.receipt is not None and not recovered.slot_held
        _independent(recovered.receipt, request)
    counts.counting = False
    assert writes == ["writer"]
    assert counts.hashed == [Path(request.target).name], counts.hashed
    assert counts.inspected == [], counts.inspected


@pytest.mark.parametrize("edited", (False, True))
def test_a_retry_with_a_recorded_post_rename_revision_compares_it_exactly(tmp_path, monkeypatch, edited):
    kind = AnalysisArtifactKind.STITCH_1D
    request = _request(tmp_path / f"retry-recorded-{edited}.nexus", kind)
    output = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator())
    # The terminal's own fence fails once, after the post-rename revision is recorded.
    failed = _fail_first_post_rename_check(monkeypatch, request, "revalidate_stream_terminal")
    with pytest.raises(AnalysisArtifactCleanupPending):
        output.publish(_writer(request))
    assert failed and output.snapshot.published and output.snapshot.receipt is None
    if edited:
        _edit_in_place_keeping_size_and_mtime(Path(request.target))  # ctime moves: the exact compare refuses
    counts = _Counts(monkeypatch)
    counts.counting = True
    if edited:
        with pytest.raises(AnalysisArtifactCleanupPending) as pending:
            output.retry_cleanup()
        assert pending.value.snapshot.receipt is None
    else:
        recovered = output.retry_cleanup()
        counts.counting = False
        assert recovered.receipt is not None
        _independent(recovered.receipt, request)
    counts.counting = False
    # The parent rehashes the public file three times and inspects it again on this retry.
    assert counts.hashed == [] and counts.finite_hashed == [], (counts.hashed, counts.finite_hashed)
    assert counts.inspected == [], counts.inspected
