# -*- coding: utf-8 -*-
"""R1 — DirectoryIndex: immutable snapshot/candidate values and transitions.

No probing, no retry yet (that is test_directory_index_retry.py, layered on
top in the next commit) — just: does poll() produce correct added/changed/
removed/unchanged deltas, monotonic generations, and object-identity reuse on
an unchanged poll.
"""

from __future__ import annotations

from pathlib import Path

from xrd_tools.sources.directory_index import DirectoryIndex, IndexDelta, Snapshot


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.write_bytes(data)
    return path


# ---- initial state ----------------------------------------------------------


def test_empty_index_starts_at_generation_zero_with_no_candidates(tmp_path):
    index = DirectoryIndex(tmp_path)
    assert index.snapshot.generation == 0
    assert index.snapshot.candidates == ()
    assert index.last_delta.unchanged is True


def test_first_poll_reports_every_candidate_as_added(tmp_path):
    _touch(tmp_path / "a.nxs")
    _touch(tmp_path / "b.nxs")

    index = DirectoryIndex(tmp_path)
    snapshot = index.poll()

    assert snapshot.generation == 1
    assert {c.path.name for c in snapshot.candidates} == {"a.nxs", "b.nxs"}
    assert {c.path.name for c in index.last_delta.added} == {"a.nxs", "b.nxs"}
    assert index.last_delta.changed == ()
    assert index.last_delta.removed == ()
    assert index.last_delta.unchanged is False


# ---- unchanged poll: reuse, not rebuild -------------------------------------


def test_unchanged_poll_returns_the_same_snapshot_object(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path)

    first = index.poll()
    second = index.poll()

    assert second is first          # object identity: no rebuild/re-sort
    assert index.last_delta.unchanged is True


def test_unchanged_poll_keeps_generation_and_candidate_identity_stable(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path)

    first = index.poll()
    for _ in range(3):
        again = index.poll()
        assert again.generation == first.generation
        assert again.candidates == first.candidates


# ---- added / changed / removed transitions ----------------------------------


def test_added_file_produces_one_generation_bump_and_exact_delta(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path)
    index.poll()

    _touch(tmp_path / "b.nxs")
    snapshot = index.poll()

    assert snapshot.generation == 2
    assert {c.path.name for c in index.last_delta.added} == {"b.nxs"}
    assert index.last_delta.changed == ()
    assert index.last_delta.removed == ()


def test_removed_file_produces_exact_delta(tmp_path):
    a = _touch(tmp_path / "a.nxs")
    _touch(tmp_path / "b.nxs")
    index = DirectoryIndex(tmp_path)
    index.poll()

    a.unlink()
    snapshot = index.poll()

    assert snapshot.generation == 2
    assert index.last_delta.added == ()
    assert index.last_delta.changed == ()
    assert index.last_delta.removed == (a,)
    assert {c.path.name for c in snapshot.candidates} == {"b.nxs"}


def test_changed_content_produces_exact_delta(tmp_path):
    a = _touch(tmp_path / "a.nxs", data=b"one")
    index = DirectoryIndex(tmp_path)
    index.poll()

    # Force a different (size, mtime_ns) stamp.
    a.write_bytes(b"a much longer payload than before")
    snapshot = index.poll()

    assert snapshot.generation == 2
    assert index.last_delta.added == ()
    assert index.last_delta.removed == ()
    assert [c.path for c in index.last_delta.changed] == [a]


def test_one_poll_produces_exactly_one_generation_transition_for_a_mixed_delta(tmp_path):
    a = _touch(tmp_path / "a.nxs", data=b"one")
    b = _touch(tmp_path / "b.nxs")
    index = DirectoryIndex(tmp_path)
    index.poll()

    a.write_bytes(b"a much longer payload than before")
    b.unlink()
    _touch(tmp_path / "c.nxs")
    snapshot = index.poll()

    assert snapshot.generation == 2   # ONE transition, not three
    assert {c.path.name for c in index.last_delta.added} == {"c.nxs"}
    assert [c.path for c in index.last_delta.changed] == [a]
    assert index.last_delta.removed == (b,)


def test_generation_is_monotonically_increasing_across_many_polls(tmp_path):
    index = DirectoryIndex(tmp_path)
    seen = [index.snapshot.generation]
    for i in range(5):
        _touch(tmp_path / f"f{i}.nxs")
        seen.append(index.poll().generation)
    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen)   # strictly increasing, no repeats


# ---- reconfigure invalidates cleanly ----------------------------------------


def test_reconfigure_root_invalidates_the_prior_generation(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _touch(a / "x.nxs")
    _touch(b / "y.nxs")

    index = DirectoryIndex(a)
    first = index.poll()
    assert {c.path.name for c in first.candidates} == {"x.nxs"}

    changed = index.reconfigure(root=b)
    assert changed is True
    assert index.snapshot.generation == first.generation + 1
    assert index.snapshot.candidates == ()   # invalidated; not yet repolled

    second = index.poll()
    assert {c.path.name for c in second.candidates} == {"y.nxs"}
    assert second.generation == first.generation + 2


def test_reconfigure_recursive_and_name_filter_invalidate(tmp_path):
    _touch(tmp_path / "sample_bg.nxs")
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "nested.nxs")

    index = DirectoryIndex(tmp_path)
    g0 = index.poll().generation

    assert index.reconfigure(recursive=True) is True
    g1 = index.poll().generation
    assert g1 > g0
    assert {c.path.name for c in index.snapshot.candidates} == {"sample_bg.nxs", "nested.nxs"}

    assert index.reconfigure(name_filter="-bg") is True
    g2 = index.poll().generation
    assert g2 > g1
    assert {c.path.name for c in index.snapshot.candidates} == {"nested.nxs"}


def test_reconfigure_with_no_actual_change_is_a_no_op(tmp_path):
    index = DirectoryIndex(tmp_path, recursive=False, name_filter=None)
    index.poll()
    gen_before = index.snapshot.generation

    changed = index.reconfigure(root=tmp_path, recursive=False, name_filter=None)

    assert changed is False
    assert index.snapshot.generation == gen_before


def test_reconfigure_name_filter_to_none_clears_an_existing_filter(tmp_path):
    _touch(tmp_path / "sample_bg.nxs")
    _touch(tmp_path / "sample_scan.nxs")

    index = DirectoryIndex(tmp_path, name_filter="-bg")
    index.poll()
    assert {c.path.name for c in index.snapshot.candidates} == {"sample_scan.nxs"}

    assert index.reconfigure(name_filter=None) is True
    index.poll()
    assert {c.path.name for c in index.snapshot.candidates} == {
        "sample_bg.nxs", "sample_scan.nxs"}


# ---- basic type sanity -------------------------------------------------------


def test_snapshot_and_delta_are_the_documented_types(tmp_path):
    index = DirectoryIndex(tmp_path)
    assert isinstance(index.snapshot, Snapshot)
    assert isinstance(index.last_delta, IndexDelta)
