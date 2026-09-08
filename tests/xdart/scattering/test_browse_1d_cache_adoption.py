"""Focused C6b1 Browse 1-D cache lifecycle adoption oracles."""

from __future__ import annotations

from pathlib import Path
from threading import Event, current_thread, get_ident
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


def _catalog(path: Path, count: int = 3):
    from xrd_tools.io import FrameScalarCatalog, FrameScalarRow

    return FrameScalarCatalog(
        str(path.resolve()),
        "entry",
        tuple(
            FrameScalarRow(label, metadata_raw={"i0": float(label)})
            for label in range(1, count + 1)
        ),
    )


class _Reader:
    def __init__(self, path, *, resolve_source, catalog):
        assert resolve_source is False
        self.path = str(path)
        self.catalog = catalog
        self.entered = 0
        self.closed = 0

    def __enter__(self):
        self.entered += 1
        return self

    def read_scalar_catalog(self, *, cancelled):
        assert not cancelled()
        return self.catalog

    def __exit__(self, _exc_type, _exc, _tb):
        self.closed += 1


def _loader(monkeypatch, path: Path, catalog, *, open_cache):
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
    readers = []

    def open_reader(source, *, resolve_source):
        reader = _Reader(
            source, resolve_source=resolve_source, catalog=catalog,
        )
        readers.append(reader)
        return reader

    loader = module.BrowseLoader(
        open_scan=lambda _path: SimpleNamespace(entry="entry"),
        open_reader=open_reader,
        open_cache=open_cache,
    )
    return loader, readers


def test_worker_constructs_exact_empty_cache_for_651_catalog_and_detaches(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )
    from xrd_tools.io import Browse1DCache
    from xrd_tools.io.browse_1d_cache import (
        Browse1DCachePhase,
        default_browse_1d_cache_budget,
    )

    path = tmp_path / "cache-651.nexus"
    path.write_bytes(b"catalog")
    catalog = _catalog(path, 651)
    main_thread = get_ident()
    construction_threads = []
    caches = []

    def open_cache():
        construction_threads.append(current_thread().ident)
        cache = Browse1DCache()
        caches.append(cache)
        return cache

    loader, readers = _loader(
        monkeypatch, path, catalog, open_cache=open_cache,
    )
    request = BrowseLoadRequest("browse-cache-651", 1, str(path))
    loader.begin(request)
    outcome = _finish(loader, request)

    assert outcome.status is BrowseLoadStatus.READY
    assert len(construction_threads) == 1
    assert construction_threads[0] != main_thread
    assert len(caches) == 1 and len(readers) == 1
    assert (readers[0].entered, readers[0].closed) == (1, 1)
    context = loader.context_for_outcome(outcome)
    assert context is not None
    assert type(context.browse_1d_cache) is Browse1DCache
    assert context.browse_1d_cache is caches[0]
    assert context.scalar_catalog is catalog
    assert context.frame_ids is context.loaded_labels is catalog.labels
    assert caches[0].budget_bytes == default_browse_1d_cache_budget()
    assert caches[0].resident_keys == ()
    assert caches[0].resident_bytes == 0
    assert caches[0].resident_root_count == 0
    assert len(context.record_store) == 0
    assert len(context.publication_store) == 0

    assert loader.consume(outcome) is context
    receipt = loader.release_context(context)
    assert receipt.cleanup_status.value == "cleaned"
    assert caches[0].phase is Browse1DCachePhase.CLOSED
    assert context.browse_1d_cache is None
    assert context.scalar_catalog is None
    assert context.released


def test_worker_refuses_foreign_cache_before_context_publication(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    path = tmp_path / "foreign-cache.nexus"
    path.write_bytes(b"catalog")
    loader, _readers = _loader(
        monkeypatch,
        path,
        _catalog(path),
        open_cache=lambda: object(),
    )
    request = BrowseLoadRequest("foreign-cache", 1, str(path))
    loader.begin(request)
    outcome = _finish(loader, request)

    assert outcome.status is BrowseLoadStatus.FAILED
    assert "foreign 1-D cache" in outcome.detail
    assert loader.context_for_outcome(outcome) is None
    assert loader.consume(outcome) is None


@pytest.mark.parametrize("outcome_kind", ("cancelled", "failed"))
def test_cancelled_or_failed_context_construction_closes_exact_allocated_cache(
    tmp_path, monkeypatch, outcome_kind: str,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )
    from xdart.modules.display_context import BrowseContext
    from xrd_tools.io import Browse1DCache
    from xrd_tools.io.browse_1d_cache import Browse1DCachePhase

    path = tmp_path / f"cache-{outcome_kind}.nexus"
    path.write_bytes(b"catalog")
    catalog = _catalog(path)
    allocated = Event()
    release = Event()
    caches = []
    close_calls = []
    real_close = Browse1DCache.close

    def close(cache):
        close_calls.append(cache)
        return real_close(cache)

    def open_cache():
        cache = Browse1DCache(1 << 20)
        caches.append(cache)
        if outcome_kind == "cancelled":
            allocated.set()
            assert release.wait(2.0)
        return cache

    monkeypatch.setattr(Browse1DCache, "close", close)
    if outcome_kind == "failed":
        monkeypatch.setattr(
            BrowseContext,
            "mark_loaded",
            lambda _context: (_ for _ in ()).throw(
                RuntimeError("injected context admission")
            ),
        )
    loader, _readers = _loader(
        monkeypatch, path, catalog, open_cache=open_cache,
    )
    request = BrowseLoadRequest(
        f"browse-cache-{outcome_kind}", 1, str(path),
    )
    loader.begin(request)
    if outcome_kind == "cancelled":
        assert allocated.wait(2.0)
        pending = loader.cancel(request)
        assert pending.cleanup_status.value == "cleanup_pending"
        release.set()
    outcome = _finish(loader, request)

    expected = (
        BrowseLoadStatus.CANCELLED
        if outcome_kind == "cancelled"
        else BrowseLoadStatus.FAILED
    )
    assert outcome.status is expected
    assert len(caches) == 1
    assert close_calls == [caches[0]]
    assert caches[0].phase is Browse1DCachePhase.CLOSED
    assert loader.context_for_outcome(outcome) is None
    assert loader.consume(outcome) is None


def test_release_cut_retains_exact_cache_and_catalog_for_retry(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )
    from xdart.modules.display_context import BrowseContext
    from xrd_tools.io import Browse1DCache
    from xrd_tools.io.browse_1d_cache import Browse1DCachePhase

    path = tmp_path / "release-close.nexus"
    path.write_bytes(b"catalog")
    catalog = _catalog(path)
    caches = []

    def open_cache():
        cache = Browse1DCache(1 << 20)
        caches.append(cache)
        return cache

    loader, _readers = _loader(
        monkeypatch, path, catalog, open_cache=open_cache,
    )
    request = BrowseLoadRequest("release-close", 1, str(path))
    loader.begin(request)
    outcome = _finish(loader, request)
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert context is not None
    cache = caches[0]
    calls = []
    real_close = Browse1DCache.close
    real_detach = BrowseContext.detach_browse_1d_cache
    real_release = BrowseContext.release

    def close(owner):
        calls.append(("close", owner))
        if sum(
            kind == "close" for kind, _item in calls
        ) == 1:
            raise RuntimeError("injected close cut")
        return real_close(owner)

    def detach(owner, expected):
        assert owner is context and expected is cache
        assert cache.phase is Browse1DCachePhase.CLOSED
        calls.append(("detach", expected))
        return real_detach(owner, expected)

    def release_context(owner):
        assert owner is context
        assert owner.browse_1d_cache is None
        assert owner.scalar_catalog is catalog
        calls.append(("release", cache))
        return real_release(owner)

    monkeypatch.setattr(Browse1DCache, "close", close)
    monkeypatch.setattr(BrowseContext, "detach_browse_1d_cache", detach)
    monkeypatch.setattr(BrowseContext, "release", release_context)

    pending = loader.release_context(context)
    assert pending.cleanup_status.value == "cleanup_pending"
    assert context.invalidated and not context.loaded
    assert not context.released
    assert context.browse_1d_cache is cache
    assert context.scalar_catalog is catalog

    cleaned = loader.release_context(context)
    assert cleaned.cleanup_status.value == "cleaned"
    assert all(item is cache for _kind, item in calls)
    assert [kind for kind, _item in calls][-2:] == ["detach", "release"]
    assert cache.phase is Browse1DCachePhase.CLOSED
    assert context.browse_1d_cache is None
    assert context.scalar_catalog is None
    assert context.released


def test_context_refuses_release_or_foreign_detach_while_cache_attached(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.modules.display_context import BrowseContext, DisplayContextError
    from xrd_tools.io import Browse1DCache

    path = tmp_path / "direct-release.nexus"
    catalog = _catalog(path)
    cache = Browse1DCache(1 << 20)
    foreign = Browse1DCache(1 << 20)
    request = BrowseLoadRequest("direct-release", 1, str(path))
    context = BrowseContext(
        context_token=request.token,
        load_generation=request.load_generation,
        operation=request,
        requested_path=request.source_path,
        scan_key="direct-release",
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
        loaded_labels=catalog.labels,
    )
    context.mark_loaded()
    with pytest.raises(DisplayContextError, match="live Browse context"):
        context.detach_browse_1d_cache(foreign)
    with pytest.raises(DisplayContextError, match="closed and detached"):
        context.release()
    assert context.browse_1d_cache is cache
    assert context.scalar_catalog is catalog
    assert context.loaded and not context.invalidated and not context.released

    context.invalidate()
    with pytest.raises(DisplayContextError, match="cleanup identity"):
        context.detach_browse_1d_cache(foreign)
    cache.close()
    context.detach_browse_1d_cache(cache)
    context.release()
    foreign.close()
    assert context.released and context.scalar_catalog is None
