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


# ===========================================================================
# R1-R3 — an adapter-OWNER change (same bytes) emits a changed delta
# ===========================================================================

import contextlib

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.adapters import (
    SourceFormatAdapter, _ADAPTERS, register_adapter)
from xrd_tools.sources.probe import ProbeResult, ProbeState


@contextlib.contextmanager
def _isolated_adapters():
    saved = dict(_ADAPTERS)
    try:
        yield
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved)


def _widget_adapter(idn: str) -> SourceFormatAdapter:
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".widget",
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(ProbeState.READY, reason=idn, kind=SourceKind.TILED),
        open=lambda spec: None)


class _FakeClock:
    def __init__(self, t=0.0):
        self._t = t

    def __call__(self):
        return self._t

    def advance(self, dt):
        self._t += dt


def test_r1r3_owner_change_on_unchanged_bytes_emits_one_changed_delta(tmp_path):
    """Gate 6: registering a higher-precedence adapter flips a candidate's
    owner without changing its bytes; poll() must emit exactly one `changed`
    (not an empty delta), bump the generation by one, and clear that path's
    provisional/terminal state."""
    with _isolated_adapters():
        register_adapter(_widget_adapter("owner_a"))
        (tmp_path / "x.widget").write_bytes(b"x")
        clock = _FakeClock()
        index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
        first = index.poll()
        assert first.candidates[0].adapter_id == "owner_a"
        g1 = first.generation

        # open a provisional window on the path so we can prove it is cleared
        p = tmp_path / "x.widget"
        index.record_probe(p, ProbeResult(ProbeState.IN_PROGRESS, reason="w"))
        assert index.retry_state(p) is not None

        # flip ownership with NO byte change (last-registered external wins)
        register_adapter(_widget_adapter("owner_b"))
        second = index.poll()

        assert second.candidates[0].adapter_id == "owner_b"
        assert second.generation == g1 + 1                     # exactly one bump
        assert [c.adapter_id for c in index.last_delta.changed] == ["owner_b"]
        assert index.last_delta.added == ()
        assert index.last_delta.removed == ()
        assert index.last_delta.unchanged is False
        assert index.retry_state(p) is None                    # path state cleared


# ===========================================================================
# R1-R5 — an unchanged poll performs NO additional natural sort
# ===========================================================================

def test_r1r5_unchanged_poll_adds_no_natural_sort(tmp_path, monkeypatch):
    """Gate 8: the second (unchanged) poll must not call os_sorted at all —
    the held defect reported two sort calls across two polls."""
    import xrd_tools.sources.discover as disc

    _touch(tmp_path / "a.nxs")
    _touch(tmp_path / "b.nxs")

    calls = {"n": 0}
    real = disc.os_sorted

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(disc, "os_sorted", counting)

    index = DirectoryIndex(tmp_path)
    index.poll()                       # first poll: a change from empty -> sorts
    after_first = calls["n"]
    assert after_first >= 1

    index.poll()                       # unchanged poll: MUST add zero sorts
    assert index.last_delta.unchanged is True
    assert calls["n"] == after_first   # no additional sort

    # a real change re-enables exactly one sort
    _touch(tmp_path / "c.nxs")
    index.poll()
    assert calls["n"] == after_first + 1


def test_r1r5_unchanged_poll_still_opens_zero_hdf5_files(tmp_path, monkeypatch):
    """The R1-R5 refactor must not reintroduce content opens: initial and
    unchanged polls both open zero HDF5 files."""
    import h5py

    _touch(tmp_path / "a.nxs")
    _touch(tmp_path / "b.h5")

    opened = []
    real = h5py.File
    monkeypatch.setattr(
        h5py, "File", lambda *a, **k: (opened.append(a), real(*a, **k))[1])

    index = DirectoryIndex(tmp_path)
    index.poll()
    index.poll()
    assert opened == []


# ===========================================================================
# R1-R9 — IndexDelta queues follow natural snapshot order (not fs order)
# ===========================================================================

def test_r1r9_delta_queues_are_naturally_ordered_under_shuffled_walking(tmp_path, monkeypatch):
    """A shuffled filesystem scan must still yield naturally-ordered snapshot
    AND delta tuples — a consumer of last_delta.added/changed must never regain
    unsorted directory order (the held defect exposed fs order in delta_added)."""
    import xrd_tools.sources.discover as disc

    for n in ("scan_1.nxs", "scan_2.nxs", "scan_10.nxs"):
        _touch(tmp_path / n)

    real = disc._iter_files_unordered
    # force a reverse (non-natural) walk order to expose any unsorted delta
    monkeypatch.setattr(
        disc, "_iter_files_unordered",
        lambda d, r: sorted(real(d, r), key=lambda p: p.name, reverse=True))

    natural = ["scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]
    index = DirectoryIndex(tmp_path)
    snap = index.poll()

    assert [c.path.name for c in snap.candidates] == natural
    assert [c.path.name for c in index.last_delta.added] == natural   # NOT fs order


def test_r1r9_mixed_delta_added_and_changed_follow_snapshot_order(tmp_path, monkeypatch):
    """Added and changed queues on a mixed changed poll are both in natural
    snapshot order; removed follows prior (natural) snapshot order."""
    import xrd_tools.sources.discover as disc
    real = disc._iter_files_unordered
    monkeypatch.setattr(
        disc, "_iter_files_unordered",
        lambda d, r: sorted(real(d, r), key=lambda p: p.name, reverse=True))

    a = _touch(tmp_path / "scan_1.nxs", data=b"one")
    _touch(tmp_path / "scan_2.nxs")
    b = _touch(tmp_path / "scan_20.nxs")
    index = DirectoryIndex(tmp_path)
    index.poll()

    # change scan_1 bytes, add scan_3 + scan_10, remove scan_20
    a.write_bytes(b"a much longer body")
    _touch(tmp_path / "scan_3.nxs")
    _touch(tmp_path / "scan_10.nxs")
    b.unlink()
    index.poll()

    assert [c.path.name for c in index.last_delta.added] == ["scan_3.nxs", "scan_10.nxs"]
    assert [c.path.name for c in index.last_delta.changed] == ["scan_1.nxs"]
    assert [p.name for p in index.last_delta.removed] == ["scan_20.nxs"]
    assert [c.path.name for c in index.snapshot.candidates] == [
        "scan_1.nxs", "scan_2.nxs", "scan_3.nxs", "scan_10.nxs"]
