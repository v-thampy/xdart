"""TERMINAL-SUPPORT-128: the saved-support rebuild after a Run handoff (Claude, 2026-10-30).

His choice: "Test it as a small task (Recommended)" ("Write a direct test first. If it's real, I bring you a fix to
approve."). A Run handoff opens its output in Browse in terminal mode: the load's snapshot comes from
revalidate_stream_terminal and carries the seal's digest and no ctime. The angular-support strip's rebuild
(reconstruct_saved_support) revalidates that snapshot around its reads with revalidate_target_snapshot, whose no-ctime
path re-hashes the whole file and compares digests. For a full seal (digest = the file's SHA) that is two whole-file
hashes per job; for a held seal (digest = the checkpoint's evidence digest) it can only refuse. The control is the same
file opened in snapshot mode. Real Run output written through NexusSink (tests.core.reintegrate_support).
"""
from __future__ import annotations

import hashlib
import os

import pytest

from tests.core.reintegrate_support import _seed_existing


def _rebuild(target, snapshot, labels):
    from xrd_tools.analysis.angular_support import SavedSupportRequest, reconstruct_saved_support
    return reconstruct_saved_support(SavedSupportRequest(str(target), "entry", snapshot, labels, labels))


def _summary(profiles):
    return {label: (profile.available, profile.reason, profile.detail) for label, profile in profiles.items()}


def _full_seal(target):
    from xrd_tools.io.output_transaction import StreamTerminal
    observed = os.stat(target)
    return StreamTerminal(str(target), int(observed.st_size), hashlib.sha256(target.read_bytes()).hexdigest(), 1,
                          int(observed.st_dev), int(observed.st_ino), int(observed.st_mtime_ns),
                          int(observed.st_ctime_ns))


@pytest.fixture
def hashes(monkeypatch):
    from xrd_tools.io import output_transaction as transaction
    real, counted = transaction._sha256_handle, []

    def counting(handle):
        counted.append(os.fstat(handle.fileno()).st_size)
        return real(handle)
    monkeypatch.setattr(transaction, "_sha256_handle", counting)
    return counted


@pytest.mark.parametrize("domain", ("full-seal", "held-seal"))
def test_run_handoff_support_rebuild_matches_a_browse_open_without_rehashing(tmp_path, hashes, domain):
    from xrd_tools.io.output_transaction import capture_target_snapshot, revalidate_stream_terminal
    seeded = _seed_existing(tmp_path)
    target, labels = seeded.target.resolve(), seeded.labels
    control = _summary(_rebuild(target, capture_target_snapshot(target), labels))
    seal = _full_seal(target) if domain == "full-seal" else seeded.terminal.commit_identity
    if domain == "held-seal":
        assert seal.digest != hashlib.sha256(target.read_bytes()).hexdigest(), "a held stream seals evidence, not the SHA"
    snapshot = revalidate_stream_terminal(target, seal)
    hashes.clear()
    observed = _summary(_rebuild(target, snapshot, labels))
    print(f"TERMINAL-SUPPORT {domain}: control={control} terminal={observed} full_hashes={hashes}")
    assert observed == control, "the Run handoff's rebuild must agree with a plain Browse open of the same file"
    assert hashes == [], "the rebuild must not re-hash the whole output"


@pytest.mark.parametrize("domain", ("full-seal", "held-seal"))
def test_run_handoff_support_rebuild_refuses_a_changed_file(tmp_path, domain):
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.output_transaction import revalidate_stream_terminal
    seeded = _seed_existing(tmp_path)
    target, labels = seeded.target.resolve(), seeded.labels
    seal = _full_seal(target) if domain == "full-seal" else seeded.terminal.commit_identity
    snapshot = revalidate_stream_terminal(target, seal)
    with open(target, "ab") as handle:
        handle.write(b"\0")
    for profile in _rebuild(target, snapshot, labels).values():
        assert not profile.available and profile.reason is SupportReason.LEGACY_INSUFFICIENT
        assert "could not be read" in profile.detail
