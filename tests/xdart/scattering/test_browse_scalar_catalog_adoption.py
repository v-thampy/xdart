"""Focused C4 scalar-only Browse catalog adoption oracles."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest


def _finish(loader, request, *, timeout: float = 10.0):
    worker = loader._worker
    assert worker is not None
    worker.join(timeout)
    assert not worker.is_alive()
    outcome = loader.poll(request)
    assert outcome is not None
    return outcome


def _catalog(path: Path, *, metadata=None, artifact_path: str | None = None):
    from xrd_tools.io import FrameScalarCatalog, FrameScalarRow

    return FrameScalarCatalog(
        artifact_path or str(path.resolve()),
        "entry",
        (FrameScalarRow(3, metadata_raw=metadata or {"i0": 4.0}),),
    )


class _ScalarReader:
    def __init__(self, path, *, resolve_source, catalog, mutate=None):
        assert resolve_source is False
        self.path = str(path)
        self.catalog = catalog
        self.mutate = mutate
        self.entered = 0
        self.closed = 0

    def __enter__(self):
        self.entered += 1
        return self

    def read_scalar_catalog(self, *, cancelled):
        assert not cancelled()
        if self.mutate is not None:
            self.mutate()
        return self.catalog

    def __exit__(self, _exc_type, _exc, _tb):
        self.closed += 1


def _fake_loader(monkeypatch, path: Path, reader_factory):
    from xdart.gui.tabs.scattering.adapters import browse_loader as module

    admitted_path = path.resolve()
    monkeypatch.setattr(
        module,
        "canonical_browse_scan_key",
        lambda source: (
            path.stem
            if Path(source).resolve() == admitted_path
            else ""
        ),
    )
    monkeypatch.setattr(
        module, "read_browse_presentation", lambda _path: ({}, None),
    )
    return module.BrowseLoader(
        open_scan=lambda _path: SimpleNamespace(entry="entry"),
        open_reader=reader_factory,
    )


def test_default_browse_adopts_one_scalar_catalog_without_payload_rows(
    tmp_path, monkeypatch,
) -> None:
    from tests.core.reintegrate_support import _seed_existing
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )
    from xrd_tools.io import FrameScalarCatalog, FrameViewReader
    frame_view_module = importlib.import_module("xrd_tools.io.frame_view")
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")

    seeded = _seed_existing(tmp_path, monitor="i0")
    payload_calls: list[str] = []
    hashed = []
    real_hash = transaction_module._sha256_handle

    def record_hash(handle):
        hashed.append(handle.name)
        return real_hash(handle)

    monkeypatch.setattr(transaction_module, "_sha256_handle", record_hash)

    def forbidden(name):
        def call(*_args, **_kwargs):
            payload_calls.append(name)
            raise AssertionError(f"scalar Browse read payload through {name}")
        return call

    monkeypatch.setattr(FrameViewReader, "_view_for", forbidden("_view_for"))
    for name in ("_read_1d_row", "_read_2d_row", "_read_thumbnail"):
        monkeypatch.setattr(frame_view_module, name, forbidden(name))

    loader = module.BrowseLoader(max_items=2)
    request = BrowseLoadRequest("scalar-default", 1, str(seeded.target))
    loader.begin(request)
    outcome = _finish(loader, request, timeout=20.0)
    assert outcome.status is BrowseLoadStatus.READY
    assert payload_calls == []
    operation = loader._active
    assert operation is not None and operation.reader is None

    context = loader.consume(outcome)
    assert context is not None and context.loaded
    catalog = context.scalar_catalog
    assert type(catalog) is FrameScalarCatalog
    assert context.frame_ids is catalog.labels
    assert context.loaded_labels is catalog.labels
    assert catalog.labels == seeded.labels
    assert len(context.publication_store) == 0
    assert len(context.record_store) == 0
    assert context.frames == {}
    assert context.viewer_rows_1d == {}
    assert context.viewer_rows_2d == {}
    aggregate = context.norm_aggregate
    assert aggregate.row_count == len(seeded.labels)
    assert aggregate.revision == 1
    assert dict(aggregate.channels)["i0"] == (
        float(sum(label + 1 for label in seeded.labels)),
        len(seeded.labels),
    )

    # The initial digest binds the file; unchanged-file revalidation must not
    # reread the entire artifact for load completion or first-frame hydration.
    from xdart.gui.tabs.scattering.browse_1d_hydration import Browse1DHydrationLane
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.modules.display_context import DisplaySelection
    from tests.xdart.scattering.test_browse_1d_hydration import _drain

    selection = DisplaySelection.for_context(context, 1)
    frame = DisplayFrameKey(
        RunIdentity(1, "scalar-default"), context.scan_key,
        context.requested_path, seeded.labels[0], 1,
    )
    # Payload prohibition above covers catalog adoption only.
    monkeypatch.undo()
    monkeypatch.setattr(transaction_module, "_sha256_handle", record_hash)
    lane = Browse1DHydrationLane(context)
    try:
        assert lane.submit(selection, (frame,)) is not None
        assert _drain(lane)
        assert context.browse_1d_cache.resident_keys
    finally:
        assert lane.release()

    receipt = loader.release_context(context)
    assert receipt.cleanup_status.value == "cleaned"
    assert context.released and not context.loaded
    assert context.scalar_catalog is None
    # Holding the released context cannot retain the catalog graph; only this
    # explicit test reference now keeps it alive.
    assert catalog.rows and context.scalar_catalog is None
    assert len(hashed) == 1


@pytest.mark.parametrize("finish", ("retry", "cancel", "close", "replace"))
def test_failed_reader_entry_retains_actual_hdf_close_owner(
    tmp_path, monkeypatch, finish,
) -> None:
    import h5py
    from tests.core.reintegrate_support import _seed_existing
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest, BrowseLoadStatus
    from xdart.gui.tabs.scattering.events import CleanupStatus
    from xrd_tools.io import FrameViewReader

    path = _seed_existing(tmp_path, monitor="i0").target
    readers, handles, close_attempts = [], [], []
    admitted, proceed, allow_close = Event(), Event(), Event()
    real_enter, real_close = FrameViewReader._enter_inner, h5py.File.close
    if finish != "cancel":
        proceed.set()

    def factory(*args, **kwargs):
        reader = FrameViewReader(*args, **kwargs)
        readers.append(reader)
        return reader

    def enter(reader):
        if reader is not readers[0]:
            return real_enter(reader)
        handles.append(reader._h5)
        assert handles[0].id.valid
        admitted.set()
        assert proceed.wait(10)
        raise OSError("injected read failure after real HDF open")

    def close(handle):
        if handles and handle is handles[0]:
            close_attempts.append(handle)
            if not allow_close.is_set() and (
                finish != "retry" or len(close_attempts) == 1
            ):
                raise OSError("injected HDF close hold")
        return real_close(handle)

    monkeypatch.setattr(FrameViewReader, "_enter_inner", enter)
    monkeypatch.setattr(h5py.File, "close", close)
    loader = BrowseLoader(open_reader=factory)
    request = BrowseLoadRequest("entry-failure", 1, str(path))
    try:
        loader.begin(request)
        operation, worker = loader._active, loader._worker
        assert admitted.wait(10)
        if finish == "cancel":
            assert loader.cancel(request).cleanup_status is CleanupStatus.CLEANUP_PENDING
            assert not close_attempts and handles[0].id.valid
            proceed.set()
        worker.join(10)
        assert not worker.is_alive()
        assert len(readers) == 1
        if finish == "retry":
            assert not handles[0].id.valid
            assert operation.reader is None
            assert loader.poll(request).status is BrowseLoadStatus.FAILED
            assert loader.close(request).cleanup_status is CleanupStatus.CLEANED
            return

        assert handles[0].id.valid
        assert operation.reader is readers[0], "failed entry lost its close-retry owner"
        assert loader.owns_request(request)
        if finish == "replace":
            replacement = BrowseLoadRequest("replacement", 2, str(path))
            loader.begin(replacement)
            assert loader.poll(replacement) is None
            assert len(readers) == 1 and operation.reader is readers[0]
        else:
            cleanup = loader.cancel if finish == "cancel" else loader.close
            receipt = cleanup(request)
            assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
            assert receipt.cleanup_failures
            assert loader.owns_request(request) and operation.reader is readers[0]
        assert handles[0].id.valid
        allow_close.set()
        if finish == "replace":
            loader.poll(replacement)
            outcome = _finish(loader, replacement)
            assert outcome.status is BrowseLoadStatus.READY
            context = loader.consume(outcome)
            assert context is not None
            assert loader.release_context(context).cleanup_status is CleanupStatus.CLEANED
        else:
            assert cleanup(request).cleanup_status is CleanupStatus.CLEANED
        assert not handles[0].id.valid and operation.reader is None
        assert readers[0]._snapshot_reader_cache_state().phase.value == "closed"
        assert loader.close().cleanup_status is CleanupStatus.CLEANED
    finally:
        proceed.set()
        allow_close.set()
        worker = loader._worker
        if worker is not None:
            worker.join(10)
        loader.close()
        # Also settle the parent's lost reader on a failing pre-fix assertion.
        for reader in readers:
            reader.__exit__(None, None, None)


def test_scalar_reader_cancellation_closes_before_terminal_cleanup(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    path = tmp_path / "cancel.nexus"
    path.write_bytes(b"catalog")
    entered = Event()
    readers = []

    class Reader(_ScalarReader):
        def read_scalar_catalog(self, *, cancelled):
            entered.set()
            assert entered.wait(1.0)
            while not cancelled():
                pass
            raise InterruptedError("cancelled")

    def factory(source, *, resolve_source):
        reader = Reader(
            source,
            resolve_source=resolve_source,
            catalog=_catalog(path),
        )
        readers.append(reader)
        return reader

    loader = _fake_loader(monkeypatch, path, factory)
    request = BrowseLoadRequest("scalar-cancel", 1, str(path))
    loader.begin(request)
    assert entered.wait(2.0)
    loader.cancel(request)
    outcome = _finish(loader, request)
    assert outcome.status is BrowseLoadStatus.CANCELLED
    assert len(readers) == 1
    assert (readers[0].entered, readers[0].closed) == (1, 1)
    assert loader.cancel(request).cleanup_status.value == "cleaned"


def test_scalar_reader_fail_once_close_retries_before_ready(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    path = tmp_path / "retry-close.nexus"
    path.write_bytes(b"catalog")
    readers = []

    class Reader(_ScalarReader):
        def __exit__(self, exc_type, exc, tb):
            self.closed += 1
            if self.closed == 1:
                raise OSError("injected close cut")

    def factory(source, *, resolve_source):
        reader = Reader(
            source,
            resolve_source=resolve_source,
            catalog=_catalog(path),
        )
        readers.append(reader)
        return reader

    loader = _fake_loader(monkeypatch, path, factory)
    request = BrowseLoadRequest("scalar-close-retry", 1, str(path))
    loader.begin(request)
    outcome = _finish(loader, request)
    assert outcome.status is BrowseLoadStatus.READY
    assert len(readers) == 1 and readers[0].closed == 2
    assert loader._active is not None and loader._active.reader is None
    context = loader.consume(outcome)
    assert context is not None
    assert loader.release_context(context).cleanup_status.value == "cleaned"


@pytest.mark.parametrize("kind", ("foreign-catalog", "changed-file"))
def test_scalar_catalog_or_artifact_drift_refuses_without_context(
    tmp_path, monkeypatch, kind: str,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    path = tmp_path / f"{kind}.nexus"
    path.write_bytes(b"catalog-before")
    readers = []
    foreign = str((tmp_path / "foreign.nexus").resolve())

    def factory(source, *, resolve_source):
        reader = _ScalarReader(
            source,
            resolve_source=resolve_source,
            catalog=_catalog(
                path,
                artifact_path=(foreign if kind == "foreign-catalog" else None),
            ),
            mutate=(
                (lambda: path.write_bytes(b"catalog-after!"))
                if kind == "changed-file" else None
            ),
        )
        readers.append(reader)
        return reader

    loader = _fake_loader(monkeypatch, path, factory)
    request = BrowseLoadRequest(f"scalar-{kind}", 1, str(path))
    loader.begin(request)
    outcome = _finish(loader, request)
    assert outcome.status is BrowseLoadStatus.FAILED
    assert loader.context_for_outcome(outcome) is None
    assert len(readers) == 1 and readers[0].closed == 1
    assert loader.consume(outcome) is None


def test_context_release_is_retryable_and_detaches_catalog_last() -> None:
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.modules.display_context import BrowseContext, DisplayContextError
    from xrd_tools.io import Browse1DCache

    path = Path("/tmp/release-catalog.nexus")
    catalog = _catalog(path)

    class Owned:
        def __init__(self, *, fail_once=False):
            self.fail_once = fail_once
            self.clears = 0

        def clear(self):
            self.clears += 1
            if self.fail_once and self.clears == 1:
                raise OSError("injected store clear cut")

    publication = Owned(fail_once=True)
    records = Owned()
    cache = Browse1DCache(1 << 20)
    request = BrowseLoadRequest("release-catalog", 1, str(path))
    context = BrowseContext(
        context_token=request.token,
        load_generation=request.load_generation,
        operation=request,
        requested_path=request.source_path,
        scan_key="release-catalog",
        scan=object(),
        frame=None,
        frame_ids=catalog.labels,
        frames=Owned(),
        viewer_rows_1d=Owned(),
        viewer_rows_2d=Owned(),
        publication_store=publication,
        record_store=records,
        scalar_catalog=catalog,
        browse_1d_cache=cache,
        loaded_labels=catalog.labels,
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    with pytest.raises(DisplayContextError, match="lifecycle state"):
        context.scalar_catalog = _catalog(path)
    with pytest.raises(DisplayContextError, match="lifecycle state"):
        context.loaded = False
    with pytest.raises(DisplayContextError, match="closed and detached"):
        context.release()
    assert context.loaded and not context.invalidated

    loader = BrowseLoader()
    receipt = loader.release_context(context)
    assert receipt.cleanup_status.value == "cleanup_pending"
    assert context.scalar_catalog is catalog
    assert context.browse_1d_cache is None
    assert not context.released and not context.loaded
    assert cache.phase.value == "closed"
    receipt = loader.release_context(context)
    assert receipt.cleanup_status.value == "cleaned"
    assert context.scalar_catalog is None and context.released
    assert context.frame_ids is context.loaded_labels
    assert context.frame_ids == ()
    assert records.clears == 1
    with pytest.raises(DisplayContextError):
        context.mark_loaded()


def test_display_context_remains_import_pure() -> None:
    source = (
        Path(__file__).parents[3]
        / "src/xdart/modules/display_context.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = tuple(
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ) + tuple(
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        name.startswith((
            "h5py",
            "numpy",
            "xrd_tools.io.frame_view",
            "xrd_tools.io.browse_1d_cache",
        ))
        for name in imports
    )
    assert "FrameScalarCatalog" not in source
    assert "Browse1DCache" not in source
