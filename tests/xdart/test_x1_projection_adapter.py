"""X1 GUI-adoption Slice 1 — the one projection adapter (unit-level).

These exercise the real adapter against real ``FrameRecordStore`` /
``PublicationStore`` / ``FrameRecord`` values (the ``project_frame`` seam is the
real headless function, not a fake).  The production-wired render pin +
mutation-proof test lives in ``test_x1_projection_render_pin.py``.
"""

from __future__ import annotations

import numpy as np

from xdart.gui.tabs.static_scan.frame_projection_adapter import (
    FrameProjectionAdapter,
    ProjectionRequest,
    _PublicationBackedStoreView,
)
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_frame_view,
)
from xrd_tools.core import FrameRecord, FrameView, IntegrationResult1D
from xrd_tools.session import FrameProjection, FrameRecordStore, project_frame


def _r1d(scale=1.0, *, unit="q_A^-1"):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial, intensity=intensity, sigma=np.sqrt(intensity), unit=unit)


def _view(label=0, *, meta=None, source=("/data/raw.nxs", 0)):
    return FrameView.from_results(
        label=label,
        result_1d=_r1d(),
        result_2d=None,
        metadata_raw=dict(meta or {"i0": 42.0, "temp": 300.0}),
        source_path=source[0],
        source_frame_index=source[1],
    )


def _record_store(label=0, **kw):
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(_view(label, **kw)))
    return store


def _const(value):
    return lambda: value


# --------------------------------------------------------------------------- #
# live path: real FrameRecordStore pass-through
# --------------------------------------------------------------------------- #

def test_adapter_projects_via_record_store_value_only():
    store = _record_store(6)
    adapter = FrameProjectionAdapter(_const(store), _const(None))

    projection = adapter.project(ProjectionRequest("scanA", 6, generation=0))

    assert isinstance(projection, FrameProjection)
    # equals a direct headless projection of the same store/label
    direct = project_frame(store, 6)
    assert projection.present is True
    assert dict(projection.metadata.raw) == dict(direct.metadata.raw)
    assert projection.normalization_channels == direct.normalization_channels
    assert adapter.lookup_count == 1
    assert adapter.pinned is projection


def test_adapter_returns_none_when_no_store_resolvable():
    adapter = FrameProjectionAdapter(_const(None), _const(None))
    assert adapter.project(ProjectionRequest("scanA", 0, generation=0)) is None
    assert adapter.lookup_count == 0


# --------------------------------------------------------------------------- #
# browse path: publication-backed store view
# --------------------------------------------------------------------------- #

def test_publication_backed_view_exposes_record_and_identity():
    view = _view(3, source=("/data/loaded.nxs", 3))
    store = PublicationStore()
    store.upsert(publication_from_frame_view(view))

    backed = _PublicationBackedStoreView(store)
    record = backed.get(3)
    assert record is not None
    assert backed.get_or_hydrate(3) is record          # non-hydrating alias
    # the view forwards the publication's own source identity (path form; it
    # reconciles with the record's concrete path#index inside project_frame).
    assert backed.source_identity(3) == store.get(3).source_identity
    assert backed.source_identity(3)                   # non-empty
    assert backed.get(999) is None
    assert backed.source_identity(999) == ""


def test_adapter_falls_back_to_publication_store_for_browse():
    view = _view(4)
    publications = PublicationStore()
    publications.upsert(publication_from_frame_view(view))
    # No record store -> the adapter presents the publication-backed view.
    adapter = FrameProjectionAdapter(_const(None), _const(publications))

    projection = adapter.project(ProjectionRequest("raw", 4, generation=0))
    assert isinstance(projection, FrameProjection)
    assert projection.present is True
    assert "i0" in projection.metadata.raw
    assert adapter.lookup_count == 1


def test_scan_qualified_record_store_serves_only_its_scan():
    # X1-GUI-R2: a scan-qualified active-run store serves its own scan (record
    # wins) but NOT a browse of a different scan that reuses the label (the
    # browsed scan's publication-backed source wins).  Replaces the former
    # unconditional-preference test the review flagged.
    record_store = _record_store(5, meta={"i0": 1.0})
    record_store._xdart_scan_key = "A"                   # active run owns scan A
    publications = PublicationStore()
    publications.upsert(publication_from_frame_view(
        _view(5, meta={"i0": 999.0}, source=("/data/B.nxs", 5))))
    adapter = FrameProjectionAdapter(_const(record_store), _const(publications))

    same_scan = adapter.project(ProjectionRequest("A", 5, generation=0))
    assert same_scan.metadata.raw["i0"] == 1.0           # active store serves A

    other_scan = adapter.project(ProjectionRequest("B", 5, generation=1))
    assert other_scan.metadata.raw["i0"] == 999.0        # browse B -> publication


def test_unqualified_record_store_serves_any_request_legacy():
    # A store that declares no scan identity is unqualified (loaded-scan/test) and
    # serves any request — preserves legacy behavior.
    record_store = _record_store(5, meta={"i0": 1.0})    # no _xdart_scan_key
    publications = PublicationStore()
    publications.upsert(publication_from_frame_view(_view(5, meta={"i0": 999.0})))
    adapter = FrameProjectionAdapter(_const(record_store), _const(publications))
    projection = adapter.project(ProjectionRequest("anything", 5, generation=0))
    assert projection.metadata.raw["i0"] == 1.0


# --------------------------------------------------------------------------- #
# one lookup per generation + latest-request-wins supersession
# --------------------------------------------------------------------------- #

def test_one_lookup_per_generation_repeat_returns_pinned():
    store = _record_store(6)
    adapter = FrameProjectionAdapter(_const(store), _const(None))
    req = ProjectionRequest("s", 6, generation=2)

    first = adapter.project(req)
    second = adapter.project(req)          # identical key -> pinned, no re-lookup
    assert second is first
    assert adapter.lookup_count == 1


def test_generation_bump_triggers_new_lookup():
    store = _record_store(6)
    adapter = FrameProjectionAdapter(_const(store), _const(None))

    p0 = adapter.project(ProjectionRequest("s", 6, generation=0))
    p1 = adapter.project(ProjectionRequest("s", 6, generation=1))
    assert adapter.lookup_count == 2
    assert p0 is not None and p1 is not None


def test_stale_generation_request_is_superseded_and_does_not_repin():
    store = _record_store(6)
    adapter = FrameProjectionAdapter(_const(store), _const(None))

    current = adapter.project(ProjectionRequest("s", 6, generation=5))
    # a late request from an older generation must not overwrite the pin
    stale = adapter.project(ProjectionRequest("s", 7, generation=3))
    assert stale is None
    assert adapter.pinned is current
    assert adapter.lookup_count == 1


def test_new_selection_same_generation_relookup():
    # a different frame at the same (already-latest) generation is a real new
    # request, not a stale one -> one more lookup, new pin.
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(_view(6)))
    store.upsert(FrameRecord.from_view(_view(7)))
    adapter = FrameProjectionAdapter(_const(store), _const(None))

    a = adapter.project(ProjectionRequest("s", 6, generation=4))
    b = adapter.project(ProjectionRequest("s", 7, generation=4))
    assert a is not b
    assert adapter.lookup_count == 2
