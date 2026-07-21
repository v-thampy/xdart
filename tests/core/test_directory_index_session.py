# -*- coding: utf-8 -*-
"""H19: one persistent, value-only directory authority."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.directory_index import DirectoryIndex
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.probe import ProbeResult, ProbeState


def _write(path: Path, payload=b"x") -> Path:
    path.write_bytes(payload)
    return path


def _fake_probe(_index, candidate):
    state = (
        ProbeState.PROCESSED_OUTPUT
        if "processed" in candidate.path.name
        else ProbeState.READY
    )
    return ProbeResult(state, kind=SourceKind.NEXUS_STACK)


def test_session_persists_index_and_reuses_unchanged_probe_results(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(DirectoryIndex, "probe_candidate", _fake_probe)
    _write(tmp_path / "scan_10.nxs")
    _write(tmp_path / "scan_2.nxs")
    _write(tmp_path / "processed_1.nxs")
    _write(tmp_path / "ignore.tif")

    session = DirectoryIndexSession()
    try:
        generation = session.configure(tmp_path, suffixes=(".nxs",))
        first = session.observe()
        assert first.request_generation == generation
        assert [item.candidate.path.name for item in first.candidates] == [
            "processed_1.nxs", "scan_2.nxs", "scan_10.nxs",
        ]
        assert [item.path.name for item in first.ready_snapshot.candidates] == [
            "scan_2.nxs", "scan_10.nxs",
        ]
        assert first.content_opens == 3
        assert first.excluded_count == 1

        second = session.observe()
        assert second.content_opens == 0
        assert second.discovered_snapshot.generation == (
            first.discovered_snapshot.generation)
        assert second.ready_snapshot.candidates == first.ready_snapshot.candidates
    finally:
        session.close()


def test_session_applies_suffix_and_shared_name_filter_before_probe(
    tmp_path, monkeypatch,
):
    opened = []

    def probe(index, candidate):
        opened.append(candidate.path.name)
        return _fake_probe(index, candidate)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    for name in (
        "sample_10_master.h5", "sample_2_master.h5", "sample_bg_master.h5",
        "sample_1.nxs", "other_master.h5",
    ):
        _write(tmp_path / name)

    session = DirectoryIndexSession()
    try:
        session.configure(
            tmp_path,
            name_filter="sample -bg",
            suffixes=("_master.h5",),
        )
        observed = session.observe()
        assert [item.path.name for item in observed.ready_snapshot.candidates] == [
            "sample_2_master.h5", "sample_10_master.h5",
        ]
        assert opened == ["sample_2_master.h5", "sample_10_master.h5"]
    finally:
        session.close()


def test_changed_candidate_is_reprobed_once_then_cached(tmp_path, monkeypatch):
    calls = []

    def probe(index, candidate):
        calls.append(candidate.version_stamp)
        return _fake_probe(index, candidate)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    source = _write(tmp_path / "scan.nxs", b"old")
    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        before = session.observe()
        source.write_bytes(b"new and larger")
        changed = session.observe()
        stable = session.observe()

        assert before.content_opens == 1
        assert changed.content_opens == 1
        assert [item.path for item in changed.delta.changed] == [source]
        assert stable.content_opens == 0
        assert len(calls) == 2
        assert calls[0] != calls[1]
    finally:
        session.close()


def test_raising_probe_is_cached_as_invalid_without_poisoning_session(
    tmp_path, monkeypatch,
):
    calls = []

    def probe(_index, candidate):
        calls.append(candidate.path.name)
        if candidate.path.name == "broken.nxs":
            raise RuntimeError("adapter defect")
        return _fake_probe(_index, candidate)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    _write(tmp_path / "broken.nxs")
    ready = _write(tmp_path / "ready.nxs")
    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        first = session.observe()
        second = session.observe()

        assert [item.path for item in first.ready_snapshot.candidates] == [ready]
        broken = next(
            item for item in first.candidates
            if item.candidate.path.name == "broken.nxs")
        assert broken.result.state is ProbeState.INVALID
        assert "adapter defect" in (broken.result.reason or "")
        assert second.content_opens == 0
        assert calls == ["broken.nxs", "ready.nxs"]
    finally:
        session.close()


def test_observations_are_value_only(tmp_path, monkeypatch):
    monkeypatch.setattr(DirectoryIndex, "probe_candidate", _fake_probe)
    _write(tmp_path / "scan.nxs")
    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        observed = session.observe()
        item = observed.candidates[0]
        assert set(item.__slots__) == {"candidate", "result"}
        assert not hasattr(item, "source")
        assert not hasattr(item, "handle")
        assert not hasattr(observed, "index")
    finally:
        session.close()


def test_delta_is_scoped_to_visible_suffix_and_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(DirectoryIndex, "probe_candidate", _fake_probe)
    visible = _write(tmp_path / "sample.nxs")
    excluded = _write(tmp_path / "other.nxs")
    session = DirectoryIndexSession()
    try:
        session.configure(
            tmp_path, name_filter="sample", suffixes=(".nxs",))
        session.observe()
        excluded.unlink()
        unchanged_visible = session.observe()
        assert unchanged_visible.delta.unchanged is True

        visible.unlink()
        removed = session.observe()
        assert removed.delta.removed == (visible,)
    finally:
        session.close()


def test_unchanged_provisional_candidate_is_retried_until_ready(
    tmp_path, monkeypatch,
):
    calls = 0

    def probe(index, candidate):
        nonlocal calls
        calls += 1
        state = ProbeState.IN_PROGRESS if calls == 1 else ProbeState.READY
        return index.record_probe(
            candidate, ProbeResult(state, kind=SourceKind.NEXUS_STACK))

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    source = _write(tmp_path / "writing.nxs")
    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        provisional = session.observe()
        ready = session.observe()
        idle = session.observe()

        assert provisional.pending_count == 1
        assert provisional.ready_snapshot.candidates == ()
        assert ready.ready_snapshot.candidates[0].path == source
        assert ready.content_opens == 1
        assert idle.content_opens == 0
        assert calls == 2
    finally:
        session.close()


def test_real_nascent_nexus_shell_becomes_ready_after_detector_arrives(tmp_path):
    source = tmp_path / "scan.nxs"
    with h5py.File(source, "w") as handle:
        handle.create_group("entry")

    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        provisional = session.observe()
        assert provisional.pending_count == 1
        assert provisional.ready_snapshot.candidates == ()

        with h5py.File(source, "a") as handle:
            detector = handle["entry"].create_group("instrument/detector")
            detector.create_dataset(
                "data", data=np.zeros((2, 3, 4), dtype=np.uint16))

        ready = session.observe()
        assert [item.path for item in ready.delta.changed] == [source]
        assert ready.ready_snapshot.candidates[0].path == source
        assert ready.result_for(source).state is ProbeState.READY
    finally:
        session.close()


def test_reconfigure_replaces_root_filter_and_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(DirectoryIndex, "probe_candidate", _fake_probe)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _write(first_root / "old_scan.nxs")
    keep = _write(second_root / "keep_scan.nxs")
    _write(second_root / "drop_scan.nxs")

    session = DirectoryIndexSession()
    try:
        first_generation = session.configure(
            first_root, suffixes=(".nxs",))
        first = session.observe()
        second_generation = session.configure(
            second_root, name_filter="keep", suffixes=(".nxs",))
        second = session.observe()

        assert second_generation > first_generation
        assert first.request_generation == first_generation
        assert second.request_generation == second_generation
        assert second.discovered_snapshot.root == second_root
        assert second.ready_snapshot.candidates[0].path == keep
        assert all(item.path.parent == second_root
                   for item in second.discovered_snapshot.candidates)
    finally:
        session.close()


def test_eiger_master_suffix_excludes_data_sidecars_before_probe(
    tmp_path, monkeypatch,
):
    opened = []

    def probe(index, candidate):
        opened.append(candidate.path.name)
        return _fake_probe(index, candidate)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    master = _write(tmp_path / "scan_2_master.h5")
    _write(tmp_path / "scan_2_data_000001.h5")
    _write(tmp_path / "scan_10_data_000001.h5")
    master_10 = _write(tmp_path / "scan_10_master.h5")

    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=("_master.h5",))
        observation = session.observe()
        assert [item.path for item in observation.ready_snapshot.candidates] == [
            master, master_10,
        ]
        assert opened == ["scan_2_master.h5", "scan_10_master.h5"]
    finally:
        session.close()


def test_probe_work_is_bounded_and_unseen_siblings_precede_retries(
    tmp_path, monkeypatch,
):
    calls = []

    def probe(index, candidate):
        calls.append(candidate.path.name)
        attempts = calls.count(candidate.path.name)
        state = (
            ProbeState.IN_PROGRESS
            if candidate.path.name == "scan_0.nxs" and attempts == 1
            else ProbeState.READY
        )
        return index.record_probe(
            candidate, ProbeResult(state, kind=SourceKind.NEXUS_STACK))

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", probe)
    sources = tuple(_write(tmp_path / f"scan_{index}.nxs") for index in range(6))
    session = DirectoryIndexSession(max_probes_per_observation=2)
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        first = session.observe()
        second = session.observe()
        third = session.observe()
        fourth = session.observe()
        settled = session.observe()

        assert first.content_opens == 2
        assert first.pending_count == 5
        assert [item.path for item in first.ready_snapshot.candidates] == [
            sources[1],
        ]
        assert calls[:6] == [f"scan_{index}.nxs" for index in range(6)]
        assert second.content_opens == 2
        assert third.content_opens == 2
        assert fourth.content_opens == 1
        assert [item.path for item in fourth.ready_snapshot.candidates] == list(
            sources)
        assert settled.content_opens == 0
        assert settled.pending_count == 0
    finally:
        session.close()


def test_probe_batch_configuration_rejects_unbounded_values():
    for kwargs in (
        {"max_probes_per_observation": 0},
        {"probe_time_budget_s": 0},
    ):
        try:
            DirectoryIndexSession(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid probe budget: {kwargs}")
