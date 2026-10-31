"""PERF-BROWSE-OPEN-128 stage A: what a read-only revision seal may and may not stand in for (Claude, 2026-10-30).

His words: "Open a perf task (Recommended)"; on the Reintegrate trade: "that's fine". Codex's design review 2 (A3, A4,
O2, O3): Reintegrate after a revision load takes a fresh full digest and refuses a changed revision before any read
that can lead to a write; a digest-less revision never becomes a writer's expected snapshot; the angular-coverage
keys tell two revisions apart even where TargetSnapshot equality cannot. Real Run outputs (reintegrate_support).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import threading

import pytest

from tests.core.reintegrate_support import _seed_existing


def test_reintegrate_after_a_revision_load_plans_from_a_fresh_full_digest(tmp_path):
    from xrd_tools.io.output_transaction import capture_target_revision
    from xrd_tools.reduction.reintegrate import ReintegratePlan
    seeded = _seed_existing(tmp_path)
    revision = capture_target_revision(seeded.target)
    plan = ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation,
        source_root=seeded.target.parent, expected_target_snapshot=revision, expected_labels=seeded.labels)
    assert plan.expected_target_snapshot.digest == hashlib.sha256(seeded.target.read_bytes()).hexdigest()
    assert plan.expected_target_snapshot.ctime_ns == revision.ctime_ns


def test_reintegrate_refuses_a_changed_revision_before_reading_the_output(tmp_path, monkeypatch):
    from xrd_tools.io.output_transaction import capture_target_revision
    from xrd_tools.reduction import reintegrate as module
    seeded = _seed_existing(tmp_path)
    revision = capture_target_revision(seeded.target)
    with open(seeded.target, "ab") as handle:
        handle.write(b"\0")
    effects = []
    monkeypatch.setattr(module, "_inspect_artifact", lambda *a, **k: effects.append("inspect"))
    monkeypatch.setattr(module, "_prepare_gi_scouts", lambda *a, **k: effects.append("scout"))
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        module.ReintegratePlan.from_artifact(
            seeded.target, entry="entry", dimension="1d", preparation=seeded.preparation,
            source_root=seeded.target.parent, expected_target_snapshot=revision, expected_labels=seeded.labels)
    assert effects == []


def test_a_digest_less_revision_never_becomes_a_writers_expected_snapshot(tmp_path):
    """Preservation: a writer compares its expected snapshot with a fresh full capture, so a revision (no digest)
    handed to it refuses rather than standing in for the content seal."""
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import NexusSink, ReductionPlan, Scan
    from tests.core.test_vnext_p34_existing_replacement import _replacement_source_context
    seeded = _seed_existing(tmp_path)
    revision = replace(capture_target_snapshot(seeded.target), digest=None)
    sink = NexusSink.for_existing_replacement(
        seeded.target, expected_target_snapshot=revision, dimension="1d", labels=seeded.labels, audit_bytes=b"{}",
        selected_plan=seeded.preparation["selected_plan"]["bai_args"], selected_gi_mode=None,
        source_base=seeded.target.parent, file_lock=threading.RLock(), flush_every=None,
        **_replacement_source_context(seeded.target))
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        sink.begin(Scan("revision", []), ReductionPlan(integration_2d=None))


def test_angular_coverage_keys_tell_two_revisions_apart(tmp_path):
    """Two revisions of one path with equal size, mtime and inode compare equal as TargetSnapshots (ctime is
    compare=False; a revision has no digest), so a key built on snapshot equality could serve the old file's coverage."""
    from xrd_tools.analysis.angular_support import SavedSupportRequest
    from xrd_tools.io.output_transaction import TargetSnapshot
    first = TargetSnapshot(True, 8, 1_000, 1, 2, None, 10)
    second = TargetSnapshot(True, 8, 1_000, 1, 2, None, 11)
    assert first == second
    labels = (1, 2)
    assert SavedSupportRequest("o.nexus", "entry", first, labels, labels).key != \
        SavedSupportRequest("o.nexus", "entry", second, labels, labels).key
