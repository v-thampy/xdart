"""PERF-BROWSE-OPEN-128 stage B: Browse reads per-frame facts only for the frames it shows.

Frozen before the product change (design section 8, rows O4b-O4f). Real writer, loader, 1-D lane and projection;
only the facts codec is observed.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from threading import Event, current_thread, main_thread

import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_browse_1d_hydration import _drain
from tests.xdart.scattering.test_browse_scalar_catalog_adoption import _finish


def _store_type():
    return import_module("xrd_tools.io").FrameFactsStore


def _facts_types():
    io = import_module("xrd_tools.io")
    return io.FrameFacts, io.FrameFactsFailure


def _seed(tmp_path):
    from tests.core.reintegrate_support import _seed_existing

    return _seed_existing(tmp_path, monitor="i0")


def _open(target):
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest

    loader = module.BrowseLoader(max_items=2)
    request = BrowseLoadRequest("frame-facts", 1, str(target))
    loader.begin(request)
    outcome = _finish(loader, request, timeout=20.0)
    return loader, outcome


def _scope(context):
    from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xdart.modules.display_context import DisplaySelection

    identity = RunIdentity(1, "frame-facts")
    frames = tuple(
        DisplayFrameKey(identity, context.scan_key, context.requested_path, label, ordinal)
        for ordinal, label in enumerate(context.frame_ids, 1)
    )
    navigation = FrameNavigationProjection(frames, frames[0], frames)
    selection = DisplaySelection.for_context(context, 1)
    return Browse1DHydrationLane(context), selection, navigation, {frame.local_frame_label: frame for frame in frames}


def _project(context, lane, selection, navigation, targets):
    from xdart.gui.tabs.scattering.browse_1d_projection import project_browse_1d

    return project_browse_1d(context, lane, selection, navigation, targets, current_selection=selection)


def _hydrate(lane, selection, targets):
    assert lane.submit(selection, targets) is not None
    _drain(lane, timeout=20.0)


def _release(loader, context, lane, outcomes=()):
    for outcome in outcomes:
        if outcome.borrow_bundle is not None:
            outcome.borrow_bundle.release()
    assert lane.release()
    loader.release_context(context)


def _full_read_facts(target, label):
    """The independent full frame reader's values for the six facts."""
    from xrd_tools.io import FrameViewReader

    with FrameViewReader(target, resolve_source=False) as reader:
        view = reader.read(label)
    geometry = view.geometry
    return {
        "source": (view.source_path, view.source_frame_index),
        "has_thumbnail": view.thumbnail is not None,
        "mask_baked": bool(view.mask_baked) if view.thumbnail is not None else False,
        "incident_angle": None if geometry is None else geometry.incident_angle,
    }


def test_browse_open_reads_no_frame_facts_and_starts_an_empty_store(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xrd_tools.io import FrameViewReader

    seeded = _seed(tmp_path)
    calls: list[str] = []
    for name in ("_scalar_catalog_frame_facts", "_scalar_catalog_geometry_for_frame"):
        real = getattr(FrameViewReader, name)

        def recording(reader, frame, *, _real=real, _name=name):
            calls.append(_name)
            return _real(reader, frame)

        monkeypatch.setattr(FrameViewReader, name, recording)
    loader, outcome = _open(seeded.target)
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    try:
        assert calls == []
        assert type(context.frame_facts) is _store_type()
        assert context.frame_facts.record_count == 0
        assert context.scalar_catalog.labels == tuple(seeded.labels)
    finally:
        loader.release_context(context)
    assert context.frame_facts is None


def test_lane_reads_facts_on_a_worker_for_its_targets_only(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import Browse1DProjectionStatus
    from xrd_tools.io import FrameViewReader

    seeded = _seed(tmp_path)
    loader, outcome = _open(seeded.target)
    context = loader.consume(outcome)
    lane, selection, navigation, by_label = _scope(context)
    label = seeded.labels[1]
    observed: list[tuple[bool, tuple[int, ...]]] = []
    real = FrameViewReader.read_frame_facts

    def observing(reader, labels, *, cancelled=None):
        observed.append((current_thread() is main_thread(), tuple(labels)))
        return real(reader, labels, cancelled=cancelled)

    monkeypatch.setattr(FrameViewReader, "read_frame_facts", observing)
    FrameFacts, _failure = _facts_types()
    projected = []
    try:
        first = _project(context, lane, selection, navigation, (by_label[label],))
        assert first.status is Browse1DProjectionStatus.INCOMPLETE
        _hydrate(lane, selection, (by_label[label],))
        assert observed == [(False, (label,))]
        facts = context.frame_facts.get(label)
        assert type(facts) is FrameFacts
        assert all(context.frame_facts.get(other) is None for other in seeded.labels if other != label)
        expected = _full_read_facts(seeded.target, label)
        assert (facts.source_path, facts.source_frame_index) == expected["source"]
        assert facts.has_thumbnail is expected["has_thumbnail"]
        assert facts.mask_baked is expected["mask_baked"]
        assert (None if facts.geometry is None else facts.geometry.incident_angle) == expected["incident_angle"]
        assert facts.averaged is False

        projected.append(_project(context, lane, selection, navigation, (by_label[label],)))
        outcome_1d = projected[-1]
        assert outcome_1d.status is Browse1DProjectionStatus.COMPLETE, outcome_1d.diagnostic
        (payload,) = outcome_1d.payloads
        assert (payload.view.source_path, payload.view.source_frame_index) == expected["source"]
        assert payload.view.mask_baked is facts.mask_baked
        assert payload.view.geometry == facts.geometry
        assert payload.averaged is False
    finally:
        _release(loader, context, lane, projected)


def test_lane_reads_facts_only_when_rows_are_already_resident(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import Browse1DProjectionStatus
    from xrd_tools.io import FrameViewReader

    seeded = _seed(tmp_path)
    loader, outcome = _open(seeded.target)
    context = loader.consume(outcome)
    lane, selection, navigation, by_label = _scope(context)
    targets = tuple(by_label[label] for label in seeded.labels[:2])
    projected = []
    try:
        _hydrate(lane, selection, targets)
        FrameFacts, _failure = _facts_types()
        assert all(type(context.frame_facts.get(frame.local_frame_label)) is FrameFacts for frame in targets)
        resident_rows = context.browse_1d_cache.resident_keys
        context.frame_facts.clear()

        def no_rows(*_args, **_kwargs):
            raise AssertionError("resident rows were read again")

        monkeypatch.setattr(FrameViewReader, "read_1d_rows", no_rows)
        incomplete = _project(context, lane, selection, navigation, targets)
        assert incomplete.status is Browse1DProjectionStatus.INCOMPLETE
        _hydrate(lane, selection, targets)
        assert all(type(context.frame_facts.get(frame.local_frame_label)) is FrameFacts for frame in targets)
        assert context.browse_1d_cache.resident_keys == resident_rows
        projected.append(_project(context, lane, selection, navigation, targets))
        assert projected[-1].status is Browse1DProjectionStatus.COMPLETE, projected[-1].diagnostic
    finally:
        _release(loader, context, lane, projected)


def _two_d_only_file(path) -> None:
    from xrd_tools.core import FrameRecord
    from xrd_tools.io import write_frame_records
    from tests.core.test_frame_scalar_catalog import _view
    from tests.core.v2_fixture_factory import current_entry

    # The writer refuses per-frame-varying mode sets, so the whole output is 2-D-only (an Int 2D or slice result).
    records = tuple(
        FrameRecord.from_view(replace(_view(label, 1.0), axis_1d=None, intensity_1d=None), mode_2d="qip_qoop")
        for label in (2, 5)
    )
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, records)
        frames = entry.create_group("frames")
        for label in (2, 5):
            source = frames.create_group(f"frame_{label:04d}").create_group("source")
            source.create_dataset("path", data=np.bytes_(f"raw_{label}.h5"))
            source.create_dataset("frame_index", data=np.int64(label + 100))


def test_a_two_d_only_frame_gets_its_facts_through_the_same_lane(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.modules.display_context import BrowseContext, DisplaySelection
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.io import Browse1DCache, FrameViewReader
    from xrd_tools.io.output_transaction import capture_target_snapshot

    path = tmp_path / "two_d_only.nexus"
    _two_d_only_file(path)
    with FrameViewReader(path, resolve_source=False) as reader:
        catalog = reader.read_scalar_catalog()
    assert all(catalog.row(label).modes_1d == () for label in (2, 5))
    assert catalog.row(5).modes_2d == ("qip_qoop",)
    FrameFacts, _failure = _facts_types()
    request = BrowseLoadRequest("two-d-only", 1, str(path.resolve()))
    context = BrowseContext(
        context_token=request.token, load_generation=request.load_generation, operation=request,
        requested_path=request.source_path, scan_key="scan", scan=object(), frame=None,
        frame_ids=catalog.labels, frames={}, viewer_rows_1d={}, viewer_rows_2d={},
        publication_store=PublicationStore(max_items=4), record_store={}, scalar_catalog=catalog,
        browse_1d_cache=Browse1DCache(1 << 20), frame_facts=_store_type()(), target_entry=catalog.entry,
        loaded_labels=catalog.labels, target_snapshot=capture_target_snapshot(path),
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    rows_read: list[tuple[int, ...]] = []
    real_rows = FrameViewReader.read_1d_rows

    def recording_rows(reader, labels, **kwargs):
        rows_read.append(tuple(labels))
        return real_rows(reader, labels, **kwargs)

    monkeypatch.setattr(FrameViewReader, "read_1d_rows", recording_rows)
    lane = Browse1DHydrationLane(context)
    frame = DisplayFrameKey(RunIdentity(1, "two-d-only"), context.scan_key, context.requested_path, 5, 2)
    try:
        _hydrate(lane, DisplaySelection.for_context(context, 1), (frame,))
        facts = context.frame_facts.get(5)
        assert type(facts) is FrameFacts
        assert (facts.source_path, facts.source_frame_index) == ("raw_5.h5", 105)
        assert rows_read == []
    finally:
        assert lane.release()
        context.browse_1d_cache.close()


def _bad_thumbnail(target, label) -> None:
    with h5py.File(target, "r+") as handle:
        frame = handle[f"entry/frames/frame_{label:04d}"]
        if "thumbnail" in frame:
            del frame["thumbnail"]
        frame.create_dataset("thumbnail", data=np.zeros((1, 300), dtype=np.uint8))


def test_a_malformed_thumbnail_opens_and_flags_only_its_frame(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import Browse1DProjectionStatus
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xrd_tools.io import FrameViewReader

    seeded = _seed(tmp_path)
    bad = seeded.labels[1]
    _bad_thumbnail(seeded.target, bad)
    loader, outcome = _open(seeded.target)
    assert outcome.status is BrowseLoadStatus.READY, outcome
    FrameFacts, _failure = _facts_types()
    context = loader.consume(outcome)
    lane, selection, navigation, by_label = _scope(context)
    targets = tuple(by_label[label] for label in seeded.labels)
    projected = []
    try:
        _hydrate(lane, selection, targets)
        flagged = context.frame_facts.get(bad)
        assert type(flagged) is FrameFacts
        assert (flagged.has_thumbnail, flagged.mask_baked) == (False, False)
        assert "thumbnail" in flagged.thumbnail_failure
        projected.append(_project(context, lane, selection, navigation, targets))
        assert projected[-1].status is Browse1DProjectionStatus.COMPLETE, projected[-1].diagnostic
        by_frame = {payload.frame_key.local_frame_label: payload for payload in projected[-1].payloads}
        assert by_frame[bad].view.thumbnail is None
        assert by_frame[bad].view.intensity_1d is not None
        # Viewing the frame reads its thumbnail through the preview reader, which reports the short reason.
        with FrameViewReader(seeded.target, resolve_source=False) as reader:
            with pytest.raises(ValueError, match="bounded thumbnail"):
                reader.read(bad)
    finally:
        _release(loader, context, lane, projected)


def _bad_source(target, label) -> None:
    with h5py.File(target, "r+") as handle:
        frame = handle[f"entry/frames/frame_{label:04d}"]
        if "source" in frame:
            del frame["source"]
        source = frame.create_group("source")
        source.create_dataset("path", data=np.bytes_(b"x" * 5000))
        source.create_dataset("frame_index", data=np.int64(1))


def test_a_malformed_source_refuses_only_projections_that_include_its_frame(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import Browse1DProjectionStatus
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus

    seeded = _seed(tmp_path)
    bad = seeded.labels[1]
    good = tuple(label for label in seeded.labels if label != bad)
    _bad_source(seeded.target, bad)
    loader, outcome = _open(seeded.target)
    assert outcome.status is BrowseLoadStatus.READY, outcome
    _facts, FrameFactsFailure = _facts_types()
    context = loader.consume(outcome)
    lane, selection, navigation, by_label = _scope(context)
    projected = []
    try:
        _hydrate(lane, selection, (by_label[bad],))
        assert type(context.frame_facts.get(bad)) is FrameFactsFailure
        refused = _project(context, lane, selection, navigation, (by_label[bad],))
        assert refused.status is Browse1DProjectionStatus.REFUSED
        assert f"frame {bad}" in refused.diagnostic
        good_targets = tuple(by_label[label] for label in good)
        _hydrate(lane, selection, good_targets)
        projected.append(_project(context, lane, selection, navigation, good_targets))
        assert projected[-1].status is Browse1DProjectionStatus.COMPLETE, projected[-1].diagnostic
    finally:
        _release(loader, context, lane, projected)


def test_a_superseded_lane_job_commits_no_facts(tmp_path, monkeypatch) -> None:
    from xrd_tools.io import FrameViewReader

    seeded = _seed(tmp_path)
    loader, outcome = _open(seeded.target)
    context = loader.consume(outcome)
    lane, selection, _navigation, by_label = _scope(context)
    first, second = seeded.labels[0], seeded.labels[2]
    entered, release = Event(), Event()
    real = FrameViewReader.read_frame_facts

    def blocking(reader, labels, *, cancelled=None):
        if tuple(labels) == (first,):
            entered.set()
            assert release.wait(20.0)
        return real(reader, labels, cancelled=cancelled)

    monkeypatch.setattr(FrameViewReader, "read_frame_facts", blocking)
    FrameFacts, _failure = _facts_types()
    try:
        assert lane.submit(selection, (by_label[first],)) is not None
        assert entered.wait(20.0)
        assert lane.submit(selection, (by_label[second],)) is not None
        release.set()
        _drain(lane, timeout=20.0)
        assert context.frame_facts.get(first) is None
        assert type(context.frame_facts.get(second)) is FrameFacts
    finally:
        release.set()
        _release(loader, context, lane)


def test_supersession_after_last_row_store_cannot_commit_old_facts(tmp_path, monkeypatch) -> None:
    """The row/facts boundary is still inside the request's ownership window."""
    seeded = _seed(tmp_path)
    loader, outcome = _open(seeded.target)
    context = loader.consume(outcome)
    lane, selection, _navigation, by_label = _scope(context)
    first, second = seeded.labels[0], seeded.labels[2]
    entered, release = Event(), Event()
    store_target = lane._store_target

    def blocked(task, rows, target):
        store_target(task, rows, target)
        if target.label == first:
            entered.set()
            assert release.wait(20.0)

    monkeypatch.setattr(lane, "_store_target", blocked)
    try:
        assert lane.submit(selection, (by_label[first],)) is not None
        assert entered.wait(20.0)
        assert lane.submit(selection, (by_label[second],)) is not None
        release.set()
        _drain(lane, timeout=20.0)
        assert context.frame_facts.get(first) is None
        assert type(context.frame_facts.get(second)) is _facts_types()[0]
    finally:
        release.set()
        _release(loader, context, lane)
