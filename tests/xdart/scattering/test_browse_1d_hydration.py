"""Focused C6b2 worker-lane ownership and sparse-repair oracles."""

from __future__ import annotations

from threading import Event, get_ident
from time import monotonic, sleep

import numpy as np
import pytest


def _readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _catalog(path, count: int):
    from xrd_tools.io import FrameScalarCatalog, FrameScalarRow

    return FrameScalarCatalog(
        str(path.resolve()),
        "entry",
        tuple(
            FrameScalarRow(label, modes_1d=("q",))
            for label in range(1, count + 1)
        ),
        axes_1d=(("q", "Q", "q_A^-1", False),),
    )


def _rows(catalog, labels: tuple[int, ...], serial: int = 0):
    from xrd_tools.core import Axis
    from xrd_tools.io import Frame1DModeRows, Frame1DRows

    axis = _readonly([0.5, 1.0, 1.5])
    intensities = tuple(
        _readonly([serial * 1000 + label, label + 1, label + 2])
        for label in labels
    )
    return Frame1DRows(
        catalog.artifact_path,
        catalog.entry,
        labels,
        (
            Frame1DModeRows(
                "q", Axis("Q", "q_A^-1", False, axis),
                labels, intensities, None,
            ),
        ),
        "q",
    )


def _scope(
    tmp_path,
    count: int,
    *,
    budget: int = 1 << 20,
    catalog_path=None,
    terminal: bool = False,
    legacy_terminal: bool = False,
):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.modules.display_context import BrowseContext, DisplaySelection
    from xrd_tools.io import Browse1DCache
    from xrd_tools.io.output_transaction import (
        StreamTerminal,
        capture_target_snapshot,
        revalidate_stream_terminal,
    )

    path = tmp_path / f"rows-{count}.nxs"
    path.write_bytes(b"stable Browse artifact")
    catalog = _catalog(path if catalog_path is None else catalog_path, count)
    cache = Browse1DCache(budget)
    target_snapshot = capture_target_snapshot(path)
    terminal_receipt = None
    if terminal and legacy_terminal:
        raise ValueError("terminal fixture mode must be singular")
    if terminal:
        observed = path.stat()
        terminal_receipt = StreamTerminal(
            str(path.resolve()),
            int(observed.st_size),
            "f" * 64,
            1,
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )
        target_snapshot = revalidate_stream_terminal(
            path, terminal_receipt,
        )
    elif legacy_terminal:
        terminal_receipt = StreamTerminal(
            str(path.resolve()),
            int(target_snapshot.size or 0),
            str(target_snapshot.digest),
            1,
        )
    request = BrowseLoadRequest(
        "browse-1d", 1, str(path.resolve()), terminal_receipt,
    )
    context = BrowseContext(
        context_token=request.token,
        load_generation=request.load_generation,
        operation=request,
        requested_path=request.source_path,
        scan_key="scan",
        scan=object(),
        frame=None,
        frame_ids=catalog.labels,
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store={},
        record_store={},
        scalar_catalog=catalog,
        browse_1d_cache=cache,
        target_entry=catalog.entry,
        loaded_labels=catalog.labels,
        target_snapshot=target_snapshot,
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    selection = DisplaySelection.for_context(context, 1)
    identity = RunIdentity(1, "browse-1d")
    frames = tuple(
        DisplayFrameKey(identity, "scan", str(path.resolve()), label, ordinal)
        for ordinal, label in enumerate(catalog.labels, 1)
    )
    return context, selection, frames, catalog, cache


def test_terminal_seal_reuses_loader_qualification_domain(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xrd_tools.io.output_transaction import capture_target_snapshot

    context, selection, frames, catalog, cache = _scope(
        tmp_path, 2, terminal=True,
    )
    # The synthetic modern terminal deliberately carries writer evidence that
    # is not the ordinary full-file digest.  Hydration must use the same
    # object-revision verifier as the loader rather than crossing domains.
    assert capture_target_snapshot(context.requested_path) != (
        context.target_snapshot
    )
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)

    assert lane.submit(selection, frames) is not None
    assert _drain(lane)
    assert factory.read_labels == [catalog.labels]
    assert lane.terminal_diagnostic(selection, frames) is None
    assert lane.release()
    cache.close()

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    context, selection, frames, catalog, cache = _scope(
        legacy_root, 2, legacy_terminal=True,
    )
    assert capture_target_snapshot(context.requested_path) == (
        context.target_snapshot
    )
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)
    assert lane.submit(selection, frames) is not None
    assert _drain(lane)
    assert factory.read_labels == [catalog.labels]
    assert lane.release()
    cache.close()


class _Reader:
    def __init__(self, factory, labels_block=None):
        self.factory = factory
        self.labels_block = labels_block
        self.entered = False
        self.closed = False

    def __enter__(self):
        self.entered = True
        self.factory.enter_threads.append(get_ident())
        return self

    def read_1d_rows(self, labels, *, cancelled):
        self.factory.read_labels.append(labels)
        self.factory.read_started.set()
        if self.labels_block is not None:
            assert self.labels_block.wait(2.0)
        self.factory.cancel_observations.append(cancelled())
        return _rows(self.factory.catalog, labels, len(self.factory.read_labels))

    def __exit__(self, _exc_type, _exc, _tb):
        if not self.factory.allow_close.is_set():
            self.factory.close_failures += 1
            raise RuntimeError("held reader close")
        self.closed = True
        self.factory.close_count += 1


class _ReaderFactory:
    def __init__(self, catalog):
        self.catalog = catalog
        self.read_labels = []
        self.enter_threads = []
        self.cancel_observations = []
        self.read_started = Event()
        self.block = None
        self.allow_close = Event()
        self.allow_close.set()
        self.close_count = 0
        self.close_failures = 0

    def __call__(
        self, path, *, entry, include_thumbnail, resolve_source,
    ):
        assert path == self.catalog.artifact_path
        assert entry == self.catalog.entry
        assert include_thumbnail is False
        assert resolve_source is False
        return _Reader(self, self.block)


def _drain(lane, *, timeout: float = 10.0) -> bool:
    deadline = monotonic() + timeout
    repaint = False
    while monotonic() < deadline:
        repaint = lane.consume_repaint() or repaint
        if not lane.polling_needed():
            return repaint
        sleep(0.001)
    raise AssertionError("Browse 1-D hydration did not settle")


def _wait_worker_idle(lane, *, timeout: float = 10.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        lane._progress()
        with lane._lock:
            if lane._active is None and lane._queued is None:
                return
        sleep(0.001)
    raise AssertionError("Browse 1-D hydration worker did not become idle")


def _take_completions(lane):
    from queue import Empty

    completions = []
    while True:
        try:
            completions.append(lane._completions.get_nowait())
        except Empty:
            return tuple(completions)


def test_651_labels_read_once_then_fully_resident_avoids_io(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    context, selection, frames, catalog, cache = _scope(tmp_path, 651)
    factory = _ReaderFactory(catalog)
    main_thread = get_ident()
    lane = Browse1DHydrationLane(context, open_reader=factory)

    assert lane.submit(selection, frames) is not None
    assert _drain(lane)
    assert factory.read_labels == [catalog.labels]
    assert len(factory.enter_threads) == 1
    assert factory.enter_threads[0] != main_thread
    assert factory.close_count == 1
    assert len(cache.resident_keys) == 2 * len(catalog.labels)
    assert context.publication_store == {}
    assert context.record_store == {}

    assert lane.submit(selection, frames) is not None
    assert not _drain(lane)
    assert factory.read_labels == [catalog.labels]
    assert factory.close_count == 1
    assert lane.release()
    cache.close()


def test_partial_repair_substitutes_exact_borrowed_survivor(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xrd_tools.io import browse_1d_row_name

    context, selection, frames, catalog, cache = _scope(
        tmp_path, 1, budget=4 * 3 * np.dtype(np.float64).itemsize,
    )
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)
    lane.submit(selection, frames)
    assert _drain(lane)

    axis_name = browse_1d_row_name("q", "axis")
    intensity_name = browse_1d_row_name("q", "intensity")
    outside = cache.borrow(1, 1, axis_name)
    survivor = outside.array
    competitor = _readonly(np.arange(9, dtype=np.float64))
    assert cache.begin_store(
        99, 99, (("competitor", competitor),),
    ).run() == "accepted"
    assert tuple(key.name for key in cache.resident_keys) == (
        axis_name, "competitor",
    )

    lane.submit(selection, frames)
    assert _drain(lane)
    assert factory.read_labels == [(1,), (1,)]
    with cache.borrow(1, 1, axis_name) as axis:
        assert axis.array is survivor
    with cache.borrow(1, 1, intensity_name) as intensity:
        assert intensity.array is not survivor
    assert tuple(key.name for key in cache.resident_keys) == (
        axis_name, intensity_name,
    )
    assert not outside.released
    outside.release()
    assert lane.release()
    cache.close()


def test_one_active_keeps_only_latest_queued_and_stale_has_no_repaint(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    context, selection, frames, catalog, cache = _scope(tmp_path, 3)
    factory = _ReaderFactory(catalog)
    release_first = Event()
    factory.block = release_first
    lane = Browse1DHydrationLane(context, open_reader=factory)
    first = lane.submit(selection, (frames[0],))
    assert first is not None and factory.read_started.wait(2.0)
    second = lane.submit(selection, (frames[1],))
    third = lane.submit(selection, (frames[2],))
    assert second is not None and third is not None and second is not third
    factory.block = None
    release_first.set()
    assert _drain(lane)
    assert factory.read_labels == [(1,), (3,)]
    assert (2, 2) not in {
        (key.frame, key.label) for key in cache.resident_keys
    }
    assert (3, 3) in {
        (key.frame, key.label) for key in cache.resident_keys
    }
    assert lane.release()
    cache.close()


def test_cancelled_a_then_b_then_fresh_a_runs_only_latest_a(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    context, selection, frames, catalog, cache = _scope(tmp_path, 2)
    factory = _ReaderFactory(catalog)
    release_first = Event()
    factory.block = release_first
    lane = Browse1DHydrationLane(context, open_reader=factory)

    first_a = lane.submit(selection, (frames[0],))
    assert first_a is not None and factory.read_started.wait(2.0)
    request_b = lane.submit(selection, (frames[1],))
    latest_a = lane.submit(selection, (frames[0],))
    assert request_b is not None
    assert latest_a is not None and latest_a is not first_a
    assert latest_a is not request_b

    factory.block = None
    release_first.set()
    _wait_worker_idle(lane)
    assert lane.consume_repaint()
    assert not lane.consume_repaint()
    assert factory.read_labels == [(1,), (1,)]
    coordinates = {
        (key.frame, key.label) for key in cache.resident_keys
    }
    assert (1, 1) in coordinates
    assert (2, 2) not in coordinates
    assert not lane.polling_needed()
    assert lane.release()
    cache.close()


def test_rolled_back_cache_operation_is_clean_terminal_failure(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
        Browse1DHydrationStatus,
    )
    from xrd_tools.io import Browse1DCacheOperation

    context, selection, frames, catalog, cache = _scope(tmp_path, 1)
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)
    operations = []

    def roll_back(operation):
        operations.append(operation)
        return operation.rollback()

    monkeypatch.setattr(Browse1DCacheOperation, "run", roll_back)
    assert lane.submit(selection, frames) is not None
    _wait_worker_idle(lane)
    completions = _take_completions(lane)
    assert len(completions) == 1
    assert completions[0].status is Browse1DHydrationStatus.FAILED
    assert operations and operations[0].terminal_direction == "rolled-back"
    assert cache.resident_keys == ()
    assert cache.outstanding_borrows == 0
    assert not lane.consume_repaint()
    assert not lane.polling_needed()
    assert lane.release()
    cache.close()


def test_two_label_request_fails_once_when_joint_inventory_cannot_fit(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    one_label_bytes = 2 * 3 * np.dtype(np.float64).itemsize
    context, selection, frames, catalog, cache = _scope(
        tmp_path, 2, budget=one_label_bytes,
    )
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)

    assert lane.submit(selection, frames) is not None
    assert not _drain(lane)
    assert factory.read_labels == [(1, 2)]
    assert {
        (key.frame, key.label) for key in cache.resident_keys
    } == {(1, 1)}
    assert cache.outstanding_borrows == 0
    for _ in range(3):
        assert not lane.consume_repaint()
        assert not lane.polling_needed()
    assert factory.read_labels == [(1, 2)]
    assert lane.release()
    cache.close()


@pytest.mark.parametrize("enter_mode", ("foreign", "acquire-raise"))
def test_reader_enter_failure_retains_exact_cleanup_custody(
    tmp_path, enter_mode,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    context, selection, frames, _catalog_value, cache = _scope(tmp_path, 1)

    class Reader:
        def __init__(self):
            self.acquired = False
            self.exit_calls = 0

        def __enter__(self):
            self.acquired = True
            if enter_mode == "foreign":
                return object()
            raise RuntimeError("reader acquired then raised")

        def __exit__(self, _exc_type, _exc, _tb):
            self.exit_calls += 1
            if self.exit_calls == 1:
                raise RuntimeError("reader close cut")
            self.acquired = False

    reader = Reader()
    calls = []

    def open_reader(*_args, **_kwargs):
        calls.append(reader)
        return reader

    lane = Browse1DHydrationLane(context, open_reader=open_reader)
    assert lane.submit(selection, frames) is not None
    deadline = monotonic() + 2.0
    while monotonic() < deadline:
        with lane._lock:
            task = lane._active
            if (
                task is not None
                and task.cleanup_pending
                and not task.running
            ):
                break
        sleep(0.001)
    else:
        raise AssertionError("reader cleanup custody was not retained")
    assert task.reader is reader and task.reader_entered
    assert reader.acquired and reader.exit_calls == 1

    assert not _drain(lane)
    assert calls == [reader]
    assert reader.exit_calls == 2 and not reader.acquired
    assert cache.resident_keys == ()
    assert cache.outstanding_borrows == 0
    assert lane.release()
    cache.close()


def test_catalog_artifact_mismatch_refuses_before_reader_construction(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )

    foreign = tmp_path / "foreign-catalog.nxs"
    context, _selection, _frames, _catalog_value, cache = _scope(
        tmp_path, 1, catalog_path=foreign,
    )
    calls = []

    def open_reader(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("reader constructed for a foreign catalog")

    with pytest.raises(RuntimeError, match="not exactly admitted"):
        Browse1DHydrationLane(context, open_reader=open_reader)
    assert calls == []
    assert cache.resident_keys == ()
    cache.close()


def test_thread_start_failure_is_bounded_failed_cleanup(
    tmp_path, monkeypatch,
) -> None:
    import xdart.gui.tabs.scattering.browse_1d_hydration as hydration

    context, selection, frames, _catalog_value, cache = _scope(tmp_path, 1)
    reader_calls = []

    class StartFailure:
        ident = None

        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start cut")

    monkeypatch.setattr(hydration, "Thread", StartFailure)
    lane = hydration.Browse1DHydrationLane(
        context,
        open_reader=lambda *_args, **_kwargs: reader_calls.append(True),
    )
    identity = lane.submit(selection, frames)
    assert identity is not None
    _wait_worker_idle(lane)
    completions = _take_completions(lane)
    assert len(completions) == 1
    assert completions[0].request_identity is identity
    assert completions[0].status is hydration.Browse1DHydrationStatus.FAILED
    assert reader_calls == []
    assert not lane.consume_repaint()
    assert not lane.polling_needed()
    assert lane.release()
    cache.close()


def test_thread_constructor_failure_is_bounded_failed_cleanup(
    tmp_path, monkeypatch,
) -> None:
    import xdart.gui.tabs.scattering.browse_1d_hydration as hydration

    context, selection, frames, _catalog_value, cache = _scope(tmp_path, 1)
    reader_calls = []

    def construct_failure(**_kwargs):
        raise RuntimeError("thread constructor cut")

    monkeypatch.setattr(hydration, "Thread", construct_failure)
    lane = hydration.Browse1DHydrationLane(
        context,
        open_reader=lambda *_args, **_kwargs: reader_calls.append(True),
    )
    identity = lane.submit(selection, frames)
    assert identity is not None
    _wait_worker_idle(lane)
    completions = _take_completions(lane)
    assert len(completions) == 1
    assert completions[0].request_identity is identity
    assert completions[0].status is hydration.Browse1DHydrationStatus.FAILED
    assert reader_calls == []
    assert not lane.consume_repaint()
    assert not lane.polling_needed()
    assert lane.release()
    cache.close()


def test_failed_exact_intent_projects_refused_without_resubmit(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
        project_browse_1d,
    )
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection

    context, selection, frames, _catalog_value, cache = _scope(tmp_path, 2)
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    reader_calls = []

    def read_failure(*_args, **_kwargs):
        reader_calls.append(True)
        raise RuntimeError("exact read cut")

    lane = Browse1DHydrationLane(context, open_reader=read_failure)
    first = project_browse_1d(
        context,
        lane,
        selection,
        navigation,
        frames,
        current_selection=selection,
    )
    assert first.status is Browse1DProjectionStatus.INCOMPLETE
    assert lane.submit(selection, frames) is not None
    assert not _drain(lane)
    assert reader_calls == [True]

    refused = project_browse_1d(
        context,
        lane,
        selection,
        navigation,
        frames,
        current_selection=selection,
    )
    assert refused.status is Browse1DProjectionStatus.REFUSED
    assert refused.diagnostic == "RuntimeError: exact read cut"
    assert lane.submit(selection, frames) is None
    assert reader_calls == [True]
    assert lane.terminal_diagnostic(selection, frames[:1]) is None
    assert lane.release()
    cache.close()


def test_post_read_artifact_drift_refuses_cache_publication(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xrd_tools.io.output_transaction import TargetSnapshot

    context, selection, frames, catalog, cache = _scope(tmp_path, 1)
    factory = _ReaderFactory(catalog)
    snapshots = [
        context.target_snapshot,
        TargetSnapshot(True, 999, 1, 1, 1, "changed"),
    ]
    lane = Browse1DHydrationLane(
        context,
        open_reader=factory,
        capture_snapshot=lambda _path: snapshots.pop(0),
    )
    lane.submit(selection, frames)
    assert not _drain(lane)
    assert factory.read_labels == [(1,)]
    assert cache.resident_keys == ()
    assert lane.release()
    cache.close()


def test_release_retains_exact_failed_operation_until_worker_recovery(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xrd_tools.io import Browse1DCacheOperation

    context, selection, frames, catalog, cache = _scope(tmp_path, 1)
    factory = _ReaderFactory(catalog)
    lane = Browse1DHydrationLane(context, open_reader=factory)
    allow_recovery = Event()
    recovery_attempted = Event()
    seen = []
    real_run = Browse1DCacheOperation.run

    def fail_run(operation):
        seen.append(operation)
        if allow_recovery.is_set():
            return real_run(operation)
        raise RuntimeError("injected operation run cut")

    def recover(operation):
        seen.append(operation)
        recovery_attempted.set()
        if not allow_recovery.is_set():
            raise RuntimeError("held operation recovery")
        return real_run(operation)

    monkeypatch.setattr(Browse1DCacheOperation, "run", fail_run)
    monkeypatch.setattr(Browse1DCacheOperation, "recover", recover)
    lane.submit(selection, frames)
    assert recovery_attempted.wait(2.0)
    deadline = monotonic() + 2.0
    while monotonic() < deadline:
        active = lane._active
        if active is not None and active.cleanup_pending and not active.running:
            break
        sleep(0.001)
    else:
        raise AssertionError("exact failed operation was not retained")
    operation = lane._active.operation
    assert type(operation) is Browse1DCacheOperation
    assert seen and all(item is operation for item in seen)
    assert not lane.release()
    allow_recovery.set()
    deadline = monotonic() + 2.0
    while monotonic() < deadline and not lane.release():
        sleep(0.001)
    assert lane.release()
    assert seen and all(item is operation for item in seen)
    assert len(cache.resident_keys) == 2
    cache.close()


def test_runtime_submits_only_exact_final_browse_frame_plan(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.context_runtime import _ContextRuntime

    context, _selection, _frames, _catalog_value, cache = _scope(tmp_path, 3)
    runtime = _ContextRuntime()
    selection = runtime.adopt_browse(context, context.load_request)
    owner = _BrowseHydrationOwner(context)
    original_lane = owner._one_d_lane

    class LaneSpy:
        def __init__(self):
            self.calls = []
            self.release_calls = []
            self.release_result = False

        def submit(self, submitted_selection, submitted_frames):
            self.calls.append((submitted_selection, submitted_frames))
            return object()

        def consume_repaint(self):
            return False

        def polling_needed(self):
            return False

        def release(self, **options):
            self.release_calls.append(options)
            return self.release_result

    spy = LaneSpy()
    owner._one_d_lane = spy

    class EmptyProjection:
        def resolve_browse(self, *_args, **_kwargs):
            return None

    assert runtime.project_navigation(
        EmptyProjection(), browse_hydration_owner=owner,
    ) == ()
    assert spy.calls == [
        (selection, (runtime.navigation.current,)),
    ]
    assert runtime.navigation.current is runtime.navigation.selected[0]

    class LoaderBomb:
        def release_context(self, _context):
            raise AssertionError("context/cache cleanup ran before 1-D lane")

    pending = owner.release(
        LoaderBomb(), context, preserve_pending_repaint=True,
    )
    assert pending.cleanup_status.value == "cleanup_pending"
    assert spy.release_calls == [{"preserve_pending_repaint": True}]
    assert context.browse_1d_cache is cache and context.loaded

    assert original_lane.release()
    assert owner.retire()
    cache.close()
