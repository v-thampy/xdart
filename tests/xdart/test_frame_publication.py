from __future__ import annotations

from collections import deque
from dataclasses import replace
import gc
import logging
import threading

import numpy as np
import pytest
import h5py

from xrd_tools.core import (
    Axis,
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    IntegrationResult2D,
    TwoDKind,
    assert_framerecord_equivalent,
    assert_frameview_equivalent,
)
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.session import HydrationPurpose

from xdart.modules.frame_publication import (
    FramePublication,
    PublicationStore,
    publication_error_details,
    publication_from_nexus_frame,
    publication_from_live_frame,
    publication_has_1d_errors,
    publication_has_2d_errors,
    validate_publication,
)
class DuckFrame:
    def __init__(self, *, idx=1, gi=False):
        self.idx = idx
        self.gi = gi
        self.scan_info = {"th": 0.25, "monitor": 100.0, "sample": "LaB6"}
        self.source_file = "raw_0001.tif"
        self.source_frame_idx = 0
        self.map_raw = np.arange(16, dtype=float).reshape(4, 4)
        self.thumbnail = np.arange(4, dtype=float).reshape(2, 2)
        q = np.linspace(0.5, 3.0, 6)
        self.int_1d = IntegrationResult1D(
            radial=q,
            intensity=np.linspace(10.0, 20.0, 6),
            sigma=np.ones(6),
            unit="q_A^-1",
        )
        self.int_2d = IntegrationResult2D(
            radial=np.linspace(0.5, 3.0, 4),
            azimuthal=np.linspace(-90.0, 90.0, 3),
            intensity=np.ones((4, 3)),
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )

    def _get_incident_angle(self):
        return float(self.scan_info["th"])


def _bound_gui_light_graph(
    *, rows=3, shared_coordinate=False, same_owner_coordinate=False,
    dtype=np.float64, heavy_rows=None,
):
    """Build the exact fixed GUI layout and its bound store without Qt."""
    from xrd_tools.session import (
        Light1DBufferLayout,
        Light1DLayout,
        Light1DModeData,
        Light1DModeLayout,
        Light1DRecord,
        SessionResourceAuthority,
        SessionResourceRequirements,
        acquire_light_1d_retention,
        resolve_session_policy,
    )

    dtype = np.dtype(dtype)
    requirements = SessionResourceRequirements(
        height=4,
        width=4,
        native_itemsize=2,
        modes_1d=2,
        npt_1d=4,
        sigma_1d=1,
        modes_2d=1,
        npt_rad=3,
        npt_azim=2,
    )
    allocation = resolve_session_policy(
        requirements,
        envelope_bytes=4 * 1024 ** 3,
        requests={"publication_items": rows, "record_items": rows, **({} if heavy_rows is None else {"publication_heavy_items": heavy_rows})},
        env={},
    ).allocation
    layout = Light1DLayout(
        modes=(
            Light1DModeLayout(
                "raw",
                Light1DBufferLayout(
                    4, dtype.itemsize,
                    "per-row-q" if same_owner_coordinate else "raw-q",
                    dtype.str,
                    shared=shared_coordinate,
                ),
                Light1DBufferLayout(4, dtype.itemsize, "raw-i", dtype.str),
                Light1DBufferLayout(4, dtype.itemsize, "raw-s", dtype.str),
            ),
            Light1DModeLayout(
                "bg",
                Light1DBufferLayout(
                    4, dtype.itemsize,
                    "per-row-q" if same_owner_coordinate else "bg-q",
                    dtype.str,
                    shared=shared_coordinate,
                ),
                Light1DBufferLayout(4, dtype.itemsize, "bg-i", dtype.str),
            ),
        ),
        active_mode="bg",
    )
    authority = SessionResourceAuthority.from_allocation(allocation)
    lease = acquire_light_1d_retention(
        authority,
        owner="c2p6-gui",
        generation=7,
        layout=layout,
        requested_rows=rows,
        compatibility_byte_ceiling=(
            layout.shared_bytes + rows * layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    store = PublicationStore()
    store.bind_allocation(allocation)
    store.bind_light_1d(lease)

    def build(
        label,
        *,
        source="scan.nxs#source",
        scan="scan-owner",
        publication_source=None,
        publication_scan=None,
        publication_generation=None,
        light_generation=None,
        array_dtype=None,
        array_length=4,
        light_active="bg",
        publication_active="bg",
        drop_light_mode=None,
        omit_raw_uncertainty=False,
        add_bg_uncertainty=False,
        raw_ref=None,
        heavy=True,
        reverse_bg_coordinate=False,
    ):
        use_dtype = np.dtype(array_dtype or dtype)
        raw_coordinate = np.linspace(
            0.1, 0.4, array_length, dtype=use_dtype,
        )
        bg_coordinate = (
            raw_coordinate[::-1] if reverse_bg_coordinate else raw_coordinate
        ) if same_owner_coordinate else np.linspace(
            0.2, 0.5, array_length, dtype=use_dtype,
        )
        mode_arrays = {
            "raw": (
                raw_coordinate,
                np.full(array_length, float(label) + 1.0, dtype=use_dtype),
                None if omit_raw_uncertainty else np.full(
                    array_length, 0.5, dtype=use_dtype),
            ),
            "bg": (
                bg_coordinate,
                np.full(array_length, float(label) + 2.0, dtype=use_dtype),
                (np.full(array_length, 0.25, dtype=use_dtype)
                 if add_bg_uncertainty else None),
            ),
        }
        mode_views = {
            mode: FrameView(
                label=label,
                axis_1d=Axis("Q", "q_A^-1", values=coordinate),
                intensity_1d=intensity,
                sigma_1d=uncertainty,
                metadata_raw={"sample": "LaB6"},
                metadata_numeric={"monitor": 100.0},
                source_path="raw_0001.tif",
                source_frame_index=int(label),
            )
            for mode, (coordinate, intensity, uncertainty) in mode_arrays.items()
        }
        active_view = mode_views[publication_active]
        results_2d = {}
        if heavy:
            active_view = replace(
                active_view,
                axis_2d_x=Axis("Q", "q_A^-1", values=np.arange(3.0)),
                axis_2d_y=Axis("chi", "chi_deg", values=np.arange(2.0)),
                intensity_2d=np.full((2, 3), float(label) + 3.0),
                raw=np.full((4, 4), float(label) + 4.0),
                thumbnail=np.full((2, 2), float(label) + 5.0),
                mask_baked=True,
            )
            results_2d = {"cake": active_view}
        publication = FramePublication(
            view=active_view,
            record=FrameRecord(
                label=label,
                results_1d=mode_views,
                results_2d=results_2d,
                active_mode_1d=publication_active,
                active_mode_2d="cake" if results_2d else "default",
            ),
            source_identity=(
                source if publication_source is None else publication_source
            ),
            generation=(
                store.generation
                if publication_generation is None else publication_generation
            ),
            raw_ref=raw_ref,
            raw_status="ready" if heavy else "1d-only",
            scan_key=scan if publication_scan is None else publication_scan,
        )
        light_modes = {
            mode: Light1DModeData(*values)
            for mode, values in mode_arrays.items()
            if mode != drop_light_mode
        }
        light_record = Light1DRecord(
            row_identity=label,
            generation=(
                lease.generation if light_generation is None else light_generation
            ),
            active_mode=light_active,
            modes=light_modes,
            provenance={"source_identity": source, "scan_key": scan},
        )
        return publication, light_record

    return store, lease, allocation, authority, build


def _without_1d(publication):
    view = replace(
        publication.view,
        axis_1d=None,
        intensity_1d=None,
        sigma_1d=None,
    )
    return replace(
        publication,
        view=view,
        record=FrameRecord(
            label=publication.label,
            results_2d=publication.record.results_2d,
            active_mode_2d=publication.record.active_mode_2d,
        ),
        raw_ref=None,
    )


def test_publication_from_live_frame_keeps_raw_lazy_by_default():
    frame = DuckFrame(idx=3)

    publication = publication_from_live_frame(frame, generation=2)

    assert publication.label == 3
    assert publication.generation == 2
    assert publication.raw_ref is frame
    assert publication.view.raw is None
    assert publication.view.thumbnail is not None
    assert publication.view.mask_baked
    assert publication.metadata_numeric == {"th": 0.25, "monitor": 100.0}
    assert publication.diagnostics.ok


def test_publication_from_live_frame_can_publish_1d_light_rows():
    from xdart.modules.frame_publication import _publication_has_heavy_payload

    store = PublicationStore(max_items=None, max_heavy_items=1, max_thumbnail_items=1)

    for idx in range(1, 70):
        publication = publication_from_live_frame(
            DuckFrame(idx=idx),
            include_2d=False,
            include_thumbnail=False,
            retain_raw_ref=False,
        )
        assert publication.view.has_1d
        assert not publication.view.has_2d
        assert publication.view.thumbnail is None
        assert publication.raw_ref is None
        assert publication.raw_status == "1d-only"
        assert not _publication_has_heavy_payload(publication)
        store.upsert(publication)

    resident = store.get_many(range(1, 70))
    assert len(resident) == 69
    assert all(pub.view.has_1d for pub in resident.values())
    assert list(store._heavy_labels) == []


def test_gi_dummy_publication_is_flagged_before_display_or_save():
    frame = DuckFrame(idx=4, gi=True)
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(-1.0, 1.0, 5),
        azimuthal=np.linspace(0.0, 3.0, 4),
        intensity=np.full((5, 4), -1.0),
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )

    publication = publication_from_live_frame(frame)

    assert publication.view.two_d_kind is TwoDKind.QIP_QOOP
    assert not publication.diagnostics.ok
    assert publication.diagnostics.errors_1d == ()
    assert publication.diagnostics.errors_2d
    assert publication_has_2d_errors(publication)
    assert not publication_has_1d_errors(publication)
    assert "dummy" in publication_error_details(publication, "2d")
    assert any("dummy" in msg for msg in publication.diagnostics.errors)
    with pytest.raises(ValueError, match="dummy"):
        validate_publication(publication, raise_on_error=True)


def test_publication_1d_error_classification_is_independent_from_2d():
    frame = DuckFrame(idx=5)
    frame.int_1d = IntegrationResult1D(
        radial=np.linspace(0.5, 3.0, 6),
        intensity=np.full(6, np.nan),
        unit="q_A^-1",
    )

    publication = publication_from_live_frame(frame)

    assert publication.diagnostics.errors_1d
    assert publication.diagnostics.errors_2d == ()
    assert publication_has_1d_errors(publication)
    assert not publication_has_2d_errors(publication)


def test_publication_store_is_generation_aware():
    store = PublicationStore()
    first = publication_from_live_frame(DuckFrame(idx=1), generation=99)
    stored = store.upsert(first)

    assert stored.generation == store.generation
    assert store.labels() == (1,)
    assert store.get(1) is stored

    store.clear()
    assert len(store) == 0
    assert store.generation == 1


def test_publication_store_invalidate_drops_label_and_bumps_generation():
    """Cluster B: invalidate drops a frame's recomputed entry (and its
    carry-over) so a dropped reintegrate shadow reverts to the prior canonical
    row on the next render; generation bumps so in-flight renders re-resolve."""
    store = PublicationStore()
    store.upsert(publication_from_live_frame(DuckFrame(idx=1)))
    store.upsert(publication_from_live_frame(DuckFrame(idx=2)))
    gen0 = store.generation

    store.invalidate([1])
    assert store.get(1) is None
    assert store.get(2) is not None
    assert 1 not in store._heavy_labels and 1 not in store._thumb_labels
    assert 1 not in store._carryover
    assert store.generation == gen0 + 1

    # Invalidating an absent label is a no-op (no spurious generation bump).
    gen1 = store.generation
    store.invalidate([999])
    assert store.generation == gen1


def test_publication_store_bounds_heavy_payloads_but_keeps_metadata():
    """D2 two-tier eviction: over the heavy bound, the full arrays drop
    but the THUMBNAIL survives (tier 1) so scroll-back stays paintable;
    raw_status honestly reports 'thumbnail' (or 'evicted' if the frame
    never had one)."""
    store = PublicationStore(max_heavy_items=2)
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    assert store.labels() == (1, 2, 3)
    evicted = store.get(1)
    assert evicted is not None
    assert evicted.raw_ref is None
    assert not evicted.view.has_1d
    assert not evicted.view.has_2d
    if evicted.view.thumbnail is not None:        # tier 1: thumbnail kept
        assert evicted.raw_status == "thumbnail"
    else:
        assert evicted.raw_status == "evicted"
    assert evicted.metadata_numeric["monitor"] == 100.0
    assert evicted.diagnostics.ok

    assert store.get(2).view.has_2d
    assert store.get(3).view.has_2d


def test_publication_store_heavy_window_resize_evicts_existing_payloads():
    store = PublicationStore(max_heavy_items=2, max_thumbnail_items=None)
    for idx in (1, 2):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    store.set_max_heavy_items(1)

    assert store._max_heavy_items == 1
    assert store.get(1).raw_ref is None
    assert not store.get(1).view.has_2d
    assert store.get(1).raw_status == "thumbnail"
    assert store.get(2).view.has_2d


def test_publication_store_heavy_window_does_not_evict_1d_only_rows():
    store = PublicationStore(max_heavy_items=1, max_thumbnail_items=0)
    first = DuckFrame(idx=1)
    first.int_2d = None
    first.map_raw = None
    first.thumbnail = None
    store.upsert(publication_from_live_frame(first))

    for idx in range(2, 66):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    retained = store.get(1)
    assert retained is not None
    assert retained.view.has_1d
    assert retained.raw_ref is first
    assert retained.raw_status == "missing"


def test_publication_store_thumbnail_tier_has_its_own_bound():
    """Tier 2: thumbnails outlive the heavy bound but have their own
    (larger) bound; past it the publication drops to metadata-only."""
    store = PublicationStore(max_heavy_items=1, max_thumbnail_items=2)
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    first = store.get(1)
    assert first.view.thumbnail is None           # tier 2 evicted
    assert first.raw_status == "evicted"
    assert first.metadata_numeric["monitor"] == 100.0   # metadata kept
    second = store.get(2)
    assert second.view.thumbnail is not None      # tier 1 only
    assert not second.view.has_2d


def test_get_or_hydrate_uses_registered_hydrator():
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=0)
    store.upsert(publication_from_live_frame(DuckFrame(idx=5)))
    assert not store.get(5).view.has_1d           # fully evicted

    calls = []

    def hydrator(label):
        calls.append(label)
        return publication_from_live_frame(
            DuckFrame(idx=int(label)), include_raw=True)

    store.set_hydrator(hydrator)
    fresh = store.get_or_hydrate(5)
    assert calls == [5]
    assert fresh.view.has_1d                      # rehydrated + upserted
    # a hydrated publication short-circuits (bounds permitting)
    store2 = PublicationStore()
    store2.set_hydrator(hydrator)
    store2.upsert(publication_from_live_frame(
        DuckFrame(idx=6), include_raw=True))
    calls.clear()
    assert store2.get_or_hydrate(6).view.raw is not None
    assert calls == []                            # no needless reload


def test_get_or_hydrate_rehydrates_tier1_thumbnail():
    # TIER-1 eviction (A2 fix): the payload (1D/2D/raw) is dropped but the
    # thumbnail is KEPT (semilight, raw_status="thumbnail").  get_or_hydrate MUST
    # rehydrate it.  Regression: the old guard counted the thumbnail as "heavy"
    # (_publication_has_heavy_payload) and short-circuited, so a tier-1 frame was
    # stuck on its thumbnail forever — the bug viewer_rows_1d masked until Step 8b.
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=8)
    store.upsert(publication_from_live_frame(DuckFrame(idx=7)))
    pub = store.get(7)
    assert not pub.view.has_1d                     # payload evicted...
    assert pub.view.thumbnail is not None          # ...but thumbnail kept (tier-1)
    assert pub.raw_status == "thumbnail"

    calls = []

    def hydrator(label):
        calls.append(label)
        return publication_from_live_frame(DuckFrame(idx=int(label)))

    store.set_hydrator(hydrator)
    fresh = store.get_or_hydrate(7)
    assert calls == [7]                            # tier-1 NOW rehydrates
    assert fresh.view.has_1d


def test_publication_store_can_bound_total_items():
    store = PublicationStore(max_items=2, max_heavy_items=None)
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    assert store.labels() == (2, 3)
    assert store.get(1) is None
    assert store.get(2).view.has_1d
    assert store.get(3).view.has_1d


def test_publication_store_default_total_bound():
    store = PublicationStore(max_heavy_items=None, max_thumbnail_items=None)
    for idx in range(600):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    assert len(store) == 512
    assert store.labels()[0] == 88
    assert store.get(87) is None
    assert store.get(599) is not None


def test_publication_from_nexus_frame_matches_live_style_view(tmp_path):
    frame = DuckFrame(idx=8)
    live_publication = publication_from_live_frame(frame, include_raw=False)
    path = tmp_path / "published.nxs"

    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[8],
            results_1d=[frame.int_1d],
            results_2d=[frame.int_2d],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([8], dtype=np.int64))
        scan_data.create_dataset("th", data=np.array([0.25], dtype=np.float32))
        scan_data.create_dataset("monitor", data=np.array([100.0], dtype=np.float32))
        frame_group = entry.create_group("frames/frame_0008")
        td = frame_group.create_dataset(
            "thumbnail",
            data=np.array([[0, 85], [170, 255]], dtype=np.uint8),
        )
        td.attrs["vmin"] = 0.0
        td.attrs["vmax"] = 3.0
        td.attrs["dtype"] = "uint8"

    reload_publication = publication_from_nexus_frame(str(path), 8)

    assert reload_publication.diagnostics.ok
    assert_frameview_equivalent(
        live_publication.view,
        reload_publication.view,
    )


def _cake_widget(monitor_norm=True):
    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": False,
                                 "global_mask": np.array([], dtype=int)})()
        bkg_map_raw = 0.0
        bkg_2d = 0.0

        def normalize(self, data, metadata):
            if not monitor_norm:
                return np.asarray(data, dtype=float)
            return np.asarray(data, dtype=float) / metadata.get("monitor", 1.0)
    return _Widget()


def _cake_frame(idx, *, intensity, azimuthal):
    frame = DuckFrame(idx=idx)
    frame.scan_info = {"monitor": 10.0}
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 4),
        azimuthal=np.asarray(azimuthal, dtype=float),
        intensity=np.full((4, 3), float(intensity)),
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )
    return frame


# ===================================================================== #
# Step 3: FrameRecord-backed publications (ADR-0003)
# ===================================================================== #

from xrd_tools.core import (  # noqa: E402
    DEFAULT_MODE_KEY,
    FrameRecord,
)
from xdart.modules.frame_publication import (  # noqa: E402
    publication_from_frame_view,
)
from xrd_tools.io.nexus import write_frame_records  # noqa: E402


def _gi_multimode_frame(idx=7):
    """A GI frame with two computed 1D modes and two 2D modes; the active
    results (int_1d/int_2d) ARE the q_total / qip_qoop entries (identity)."""
    f = DuckFrame(idx=idx, gi=True)
    qip = IntegrationResult1D(
        radial=np.linspace(-5.0, 5.0, 8), intensity=np.arange(8.0),
        sigma=None, unit="qip_A^-1",
    )
    f.gi_1d = {"q_total": f.int_1d, "q_ip": qip}
    qchi = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 5), azimuthal=np.linspace(-90.0, 90.0, 4),
        intensity=np.ones((5, 4)), unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    f.gi_2d = {"qip_qoop": f.int_2d, "q_chi": qchi}
    return f


def test_live_publication_carries_record_and_view_is_active_projection():
    pub = publication_from_live_frame(DuckFrame(idx=3))      # non-GI
    assert pub.record is not None
    assert pub.record.modes_1d == (DEFAULT_MODE_KEY,)
    assert pub.record.modes_2d == (DEFAULT_MODE_KEY,)
    assert_frameview_equivalent(pub.view, pub.record.active_view())


def test_non_gi_record_is_single_default_mode():
    rec = publication_from_live_frame(DuckFrame(idx=4)).record
    assert len(rec.results_1d) == 1 and len(rec.results_2d) == 1
    assert rec.active_mode_1d == DEFAULT_MODE_KEY
    assert rec.active_mode_2d == DEFAULT_MODE_KEY


def test_live_gi_record_carries_all_modes_under_canonical_keys():
    pub = publication_from_live_frame(_gi_multimode_frame())
    rec = pub.record
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    assert set(rec.modes_2d) == {"qip_qoop", "q_chi"}
    # active inferred from int_1d/int_2d identity
    assert rec.active_mode_1d == "q_total"
    assert rec.active_mode_2d == "qip_qoop"
    # .view is the active projection and matches the record's active view
    assert_frameview_equivalent(pub.view, rec.active_view())
    # the non-active modes are real, distinct results
    assert rec.view_1d("q_ip").intensity_1d.shape[0] == 8


def test_active_mode_identity_overrides_a_stale_hint():
    # passing a hint that disagrees with int_1d must NOT diverge .view/record:
    # identity (int_1d == q_total entry) wins.
    pub = publication_from_live_frame(
        _gi_multimode_frame(), active_mode_1d="q_ip", active_mode_2d="q_chi",
    )
    assert pub.record.active_mode_1d == "q_total"   # identity, not the hint
    assert pub.record.active_mode_2d == "qip_qoop"
    assert_frameview_equivalent(pub.view, pub.record.active_view())


@pytest.mark.display_logic
def test_historical_gi_result_keys_are_rejected():
    frame = _gi_multimode_frame()
    frame.gi_1d = {"qtotal": frame.int_1d}
    with pytest.raises(ValueError, match="unknown canonical 1d mode key"):
        publication_from_live_frame(frame)


def test_unowned_active_selector_cannot_name_a_result():
    frame = DuckFrame(idx=8, gi=True)
    frame.gi_1d = {}
    with pytest.raises(ValueError, match="selector .* has no owned result"):
        publication_from_live_frame(frame, active_mode_1d="q_oop")


def test_exit_angle_2d_mode_does_not_raise():
    f = DuckFrame(idx=9, gi=True)
    exit2d = IntegrationResult2D(
        radial=np.linspace(0.0, 5.0, 4), azimuthal=np.linspace(0.0, 90.0, 3),
        intensity=np.ones((4, 3)),
        unit="exit_angle_horz_deg", azimuthal_unit="exit_angle_vert_deg",
    )
    f.int_2d = exit2d
    f.gi_2d = {"exit_angles": exit2d}
    rec = publication_from_live_frame(f).record
    assert rec.modes_2d == ("exit_angles",)
    assert rec.active_mode_2d == "exit_angles"


def test_reload_publication_carries_record(tmp_path):
    """publication_from_nexus_frame reads every persisted mode into the record."""
    recs = []
    for fi in range(2):
        f = _gi_multimode_frame(idx=fi)
        recs.append(publication_from_live_frame(f).record)
    p = str(tmp_path / "mm.nexus")
    with h5py.File(p, "w") as fh:
        from xrd_tools.io.schema import (
            PROCESSED_SCHEMA_NAME,
            PROCESSED_SCHEMA_VERSION,
            SCHEMA_NAME_ATTR,
            SCHEMA_VERSION_ATTR,
        )

        entry = fh.create_group("entry")
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        write_frame_records(entry, recs)
    pub = publication_from_nexus_frame(p, 0)
    assert pub.record is not None
    assert set(pub.record.modes_1d) == {"q_total", "q_ip"}
    assert set(pub.record.modes_2d) == {"qip_qoop", "q_chi"}
    assert_frameview_equivalent(pub.view, pub.record.active_view())


def test_eviction_thins_the_record(tmp_path):
    """Tier-1/2 eviction must drop the record's non-active mode arrays, not just
    the active .view (else record-backed publications leak past max_heavy_items)."""
    store = PublicationStore(max_heavy_items=1, max_thumbnail_items=1)
    for idx in range(4):
        store.upsert(publication_from_live_frame(_gi_multimode_frame(idx=idx)))
    # oldest frames are evicted; their record must hold no heavy arrays
    from xdart.modules.frame_publication import _publication_has_heavy_payload
    evicted = store.get(0)
    assert evicted is not None
    assert not _publication_has_heavy_payload(evicted)
    rec = evicted.record
    for mv in (*rec.results_1d.values(), *rec.results_2d.values()):
        assert mv.intensity_1d is None and mv.intensity_2d is None


# ===================================================================== #
# Step 4: integration_plot_payload (full 1D parity, NOT yet wired) + #69
# ===================================================================== #

from types import SimpleNamespace  # noqa: E402


def _int_widget(*, plot_unit_text="q (Å⁻¹)", source="1d", axis="radial",
                slice_axis=None, slice_on=False, center=0.0, width=1.0,
                wavelength_m=1e-10, gi=False):
    ui = SimpleNamespace(
        plotUnit=SimpleNamespace(currentIndex=lambda: 0,
                                 currentText=lambda: plot_unit_text),
        slice=SimpleNamespace(isChecked=lambda: slice_on),
        slice_center=SimpleNamespace(value=lambda: center),
        slice_width=SimpleNamespace(value=lambda: width),
        imageUnit=SimpleNamespace(currentText=lambda: "Q-Chi"),
    )
    return SimpleNamespace(
        scan=SimpleNamespace(name="scan", gi=gi),
        _plot_axis_info=[{"source": source, "slice_axis": slice_axis, "axis": axis}],
        ui=ui,
        normalize=lambda data, md: np.asarray(data, dtype=float)
        / ((md or {}).get("monitor", 1.0) or 1.0),
        _get_wavelength=lambda ref, **_kw: wavelength_m,
    )


# ===================================================================== #
# Fork B: PublicationStore accumulates per-mode records (ADR-0003/0005)
# ===================================================================== #

from xrd_tools.core import FrameView, assert_framerecord_equivalent  # noqa: E402
from xdart.modules.frame_publication import _publication_has_heavy_payload  # noqa: E402


def _mode_pub(label, *, mode_1d=None, mode_2d=None, scale=1.0, generation=0,
              source_identity=None, scan_key=None):
    r1 = r2 = None
    if mode_1d is not None:
        r1 = IntegrationResult1D(
            radial=np.linspace(1.0, 2.0, 5), intensity=np.arange(5.0) * scale + 1,
            sigma=None, unit="q_A^-1")
    if mode_2d is not None:
        r2 = IntegrationResult2D(
            radial=np.linspace(1.0, 2.0, 4), azimuthal=np.linspace(0.0, 1.0, 3),
            intensity=np.ones((4, 3)) * scale, unit="q_A^-1", azimuthal_unit="chi_deg")
    view = FrameView.from_results(
        label=label, result_1d=r1, result_2d=r2,
        thumbnail=np.zeros((2, 2), dtype=float), metadata_raw={"monitor": 1.0})
    rec = FrameRecord.from_view(
        view, mode_1d=mode_1d or DEFAULT_MODE_KEY, mode_2d=mode_2d or DEFAULT_MODE_KEY)
    return publication_from_frame_view(
        view, record=rec, generation=generation,
        source_identity=source_identity if source_identity is not None else str(label),
        scan_key=scan_key)


def test_accumulation_same_frame_merges_modes_view_is_latest():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", scale=1.0, generation=store.generation))
    b = _mode_pub(0, mode_1d="q_ip", scale=9.0, generation=store.generation)
    store.upsert(b)
    rec = store.get(0).record
    assert set(rec.modes_1d) == {"q_total", "q_ip"}     # accumulated
    assert rec.active_mode_1d == "q_ip"                  # incoming active wins
    assert_frameview_equivalent(store.get(0).view, b.view)  # .view stays latest


def test_accumulation_cross_dimension():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.upsert(_mode_pub(0, mode_2d="qip_qoop", generation=store.generation))
    rec = store.get(0).record
    assert set(rec.modes_1d) == {"q_total"}             # 1D mode survived
    assert set(rec.modes_2d) == {"qip_qoop"}            # 2D mode added


@pytest.mark.parametrize("checkpoint", ("verified", "failed", "stale"))
def test_checkpoint_recovery_is_private_mode_exact_and_revision_qualified(
    checkpoint,
):
    from xrd_tools.session import FrameRecordStore

    publications = PublicationStore(max_heavy_items=None, max_thumbnail_items=None)
    publications.upsert(_mode_pub(7, mode_2d="q_chi"))
    publications.upsert(_mode_pub(7, mode_2d="qip_qoop"))
    records = FrameRecordStore(max_heavy_items=None)
    stored = records.upsert(publications.get(7).record)
    records.replace_projection(7)
    revisions = {("2d", "q_chi"): 1, ("2d", "qip_qoop"): 1}
    assert records._bind_checkpoint_revisions(
        7, expected=stored, revisions=revisions,
    )
    expected = stored
    if checkpoint == "stale":
        current = records.upsert(_mode_pub(7, mode_2d="q_chi", scale=9).record)
        assert current is not expected
    marked = records._mark_checkpoint_recoverable(
        7,
        expected=expected,
        revisions=revisions,
        frame_verified=checkpoint != "failed",
        thumbnail_verified=checkpoint != "failed",
    )
    assert marked is (checkpoint == "verified")

    publications.set_heavy_evictable_probe(records.can_release_heavy)
    publications.set_thumbnail_evictable_probe(records.can_release_thumbnail)
    assert records.persisted_modes(7) == frozenset()
    assert records.durable_modes(7) == frozenset()
    assert records.can_release_record(7) is False
    assert publications.evict_heavy(7) is (checkpoint == "verified")
    assert publications.evict_thumbnail(7) is (checkpoint == "verified")


def test_checkpoint_hydration_capability_is_exact_monotonic_and_revocable():
    from xrd_tools.session import FrameRecordStore

    store = FrameRecordStore(max_heavy_items=None)
    lineage, first_checkpoint, second_checkpoint = object(), object(), object()
    store._bind_checkpoint_hydration_lineage("/tmp/live.nxs", lineage)
    first, gate = store._authorize_checkpoint_hydration(first_checkpoint)
    assert first.artifact_identity == "/tmp/live.nxs"
    assert first.run_lineage is lineage
    assert first.checkpoint_identity is first_checkpoint
    assert gate.enter(first); gate.leave()

    store._revoke_checkpoint_recovery()
    assert gate.enter(first) is False
    assert store._checkpoint_hydration_authority()[0] is None
    second, same_gate = store._authorize_checkpoint_hydration(second_checkpoint)
    assert same_gate is gate and second.generation > first.generation
    assert second.checkpoint_identity is second_checkpoint
    assert gate.enter(first) is False
    assert gate.enter(second); gate.leave()
    store.clear_checkpoint_recoverable()
    assert gate.enter(second) is False


def test_unbounded_total_projection_does_not_scan_publication_order():
    class NoScan(dict):
        def __iter__(self):
            raise AssertionError("unbounded total projection scanned all labels")

    store = PublicationStore(max_items=None)
    with store._lock:
        store._items = NoScan(store._items)
        assert store._project_total_victims_locked(7) == ()


def test_accumulation_same_mode_overwrites_no_dup():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", scale=1.0, generation=store.generation))
    store.upsert(_mode_pub(0, mode_1d="q_total", scale=5.0, generation=store.generation))
    rec = store.get(0).record
    assert rec.modes_1d == ("q_total",)                 # no duplicate
    np.testing.assert_allclose(
        rec.view_1d("q_total").intensity_1d, np.arange(5.0) * 5.0 + 1)  # latest value


def test_accumulation_different_frames_independent():
    store = PublicationStore()
    store.upsert(_mode_pub(1, mode_1d="q_total", generation=store.generation))
    store.upsert(_mode_pub(2, mode_1d="q_ip", generation=store.generation))
    assert store.get(1).record.modes_1d == ("q_total",)
    assert store.get(2).record.modes_1d == ("q_ip",)


def test_accumulation_respects_heavy_bound():
    store = PublicationStore(max_heavy_items=1, max_thumbnail_items=1)
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))  # 2-mode record
    for idx in (1, 2, 3):
        store.upsert(_mode_pub(idx, mode_1d="q_total", generation=store.generation))
    evicted = store.get(0)
    assert not _publication_has_heavy_payload(evicted)   # thinned past the bound
    for mv in (*evicted.record.results_1d.values(), *evicted.record.results_2d.values()):
        assert mv.intensity_1d is None and mv.intensity_2d is None


def test_accumulation_post_clear_does_not_merge():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.clear()                                        # scan/reintegrate boundary
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=0))  # stale generation stamp
    rec = store.get(0).record
    assert rec.modes_1d == ("q_ip",)                    # no merge across the clear
    assert store.get(0).generation == store.generation


def test_accumulation_stale_incoming_generation_is_dropped():
    # Hardening (codex follow-up): a STALE incoming generation for a frame
    # already present is from a superseded epoch — DROP it (keep the current
    # entry), so old-scan data can neither splice into nor replace the live frame.
    store = PublicationStore()
    store.clear()                                        # bump to generation 1
    assert store.generation == 1
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=1))
    returned = store.upsert(_mode_pub(0, mode_1d="q_ip", generation=0))  # stale
    rec = store.get(0).record
    assert rec.modes_1d == ("q_total",)                 # stale incoming dropped
    assert returned is store.get(0)                     # upsert returned the kept entry


def test_accumulation_stale_incoming_for_new_label_is_stored():
    # A stale incoming for a NEW label (no existing entry) is still stored
    # (coerced up) — legacy/sessionless callers rely on this.
    store = PublicationStore()
    store.clear()                                        # generation 1
    store.upsert(_mode_pub(5, mode_1d="q_total", generation=0))  # stale, new label
    assert store.get(5) is not None
    assert store.get(5).record.modes_1d == ("q_total",)
    assert store.get(5).generation == store.generation  # coerced up


def test_accumulation_different_source_identity_does_not_merge():
    # Hardening (codex): a label reused across scans/files (different non-empty
    # source_identity) after a missed clear must NOT accumulate into one record.
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation,
                           source_identity="scanA"))
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation,
                           source_identity="scanB"))
    assert store.get(0).record.modes_1d == ("q_ip",)    # plain replace, no merge


def test_accumulation_missing_source_identity_does_not_merge_known_source():
    # Unknown+unknown remains a transition fallback, but unknown+known is not
    # enough evidence to splice records for a reused label.
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation,
                           source_identity=""))
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation,
                           source_identity="scanA"))
    assert store.get(0).record.modes_1d == ("q_ip",)


def test_accumulation_first_upsert_is_plain_replace_additive():
    # behavior-preservation: a fresh label is stored verbatim (today's REPLACE).
    store = PublicationStore()
    pub = _mode_pub(0, mode_1d="q_total", generation=store.generation)
    store.upsert(pub)
    assert_framerecord_equivalent(store.get(0).record, pub.record)
    assert_frameview_equivalent(store.get(0).view, pub.view)


# ===================================================================== #
# Step 6 activation: key the record under the real GI mode + carry records
# across a same-scan reintegrate so accumulation is REAL in production.
# ===================================================================== #

def test_active_mode_cannot_key_an_unowned_result_when_gi_dicts_are_empty():
    with pytest.raises(ValueError, match="selector .* has no owned result"):
        publication_from_live_frame(
            DuckFrame(idx=1, gi=True),
            active_mode_1d="q_oop", active_mode_2d="q_chi",
        )


def test_record_stays_default_when_no_active_mode_passed():
    rec = publication_from_live_frame(DuckFrame(idx=1, gi=True)).record
    assert rec.active_mode_1d == DEFAULT_MODE_KEY
    assert rec.active_mode_2d == DEFAULT_MODE_KEY


def test_view_unchanged_by_active_mode_keying():
    f = _gi_multimode_frame(idx=2)
    base = publication_from_live_frame(f)
    keyed = publication_from_live_frame(f, active_mode_1d="q_ip", active_mode_2d="q_chi")
    assert_frameview_equivalent(base.view, keyed.view)   # display surface unchanged


def test_begin_reintegrate_empties_like_clear_but_accumulates():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    g0 = store.generation
    store.begin_reintegrate()
    assert store.generation == g0 + 1          # bumped like clear (display unchanged)
    assert store.get(0) is None                # _items emptied (mid-pass blank/build)
    # re-upsert frame 0 at q_ip this pass -> merges with the carried q_total
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))
    rec = store.get(0).record
    assert set(rec.modes_1d) == {"q_total", "q_ip"}   # ACCUMULATED across the pass
    assert rec.active_mode_1d == "q_ip"


def test_begin_reintegrate_accumulates_across_three_passes():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.begin_reintegrate()
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))
    store.begin_reintegrate()
    store.upsert(_mode_pub(0, mode_1d="q_oop", generation=store.generation))
    assert set(store.get(0).record.modes_1d) == {"q_total", "q_ip", "q_oop"}


def test_begin_reintegrate_evicted_carryover_does_not_resurrect():
    # An evicted (thinned) frame carries a thinned record; the re-upsert does
    # NOT bring back its dropped modes (they rehydrate from disk instead).
    store = PublicationStore(max_heavy_items=1, max_thumbnail_items=1)
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))
    for idx in (1, 2, 3):                       # push frame 0 past the heavy bound
        store.upsert(_mode_pub(idx, mode_1d="q_total", generation=store.generation))
    store.begin_reintegrate()
    store.upsert(_mode_pub(0, mode_1d="q_oop", generation=store.generation))
    assert store.get(0).record.modes_1d == ("q_oop",)   # no resurrection of evicted modes


def test_clear_drops_carryover():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.begin_reintegrate()                  # carries frame 0
    store.clear()                              # a full reset must drop the carry-over
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))
    assert store.get(0).record.modes_1d == ("q_ip",)    # no resurrection after clear


def test_carryover_merges_across_abspath_relpath_source():
    # P2 regression (review): a live frame's source_identity is an ABSPATH while
    # the reintegrate reload uses the RELPATH of the same file; basename-
    # normalized _same_source must treat them as the SAME source so the live
    # mode is NOT dropped on the next Integrate.
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation,
                           source_identity="/data/run1/recon_0001.tif"))  # live abspath
    store.begin_reintegrate()
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation,
                           source_identity="recon_0001.tif"))             # reload relpath
    assert set(store.get(0).record.modes_1d) == {"q_total", "q_ip"}       # accumulated


def test_carryover_missing_source_identity_does_not_merge_known_source():
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation,
                           source_identity="/data/run1/recon_0001.tif"))
    store.begin_reintegrate()
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation,
                           source_identity=""))
    assert store.get(0).record.modes_1d == ("q_ip",)


def test_one_d_reintegrate_preserves_prior_2d_mode():
    # Footgun guard (review): a 1D-only reintegrate must NOT drop the 2D mode
    # accumulated in the original run.  begin_reintegrate carries the full record;
    # the 1D-only re-upsert merges the new 1D mode while the carried 2D survives.
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", mode_2d="qip_qoop",
                           generation=store.generation))
    store.begin_reintegrate()                       # a 1D Integrate pass
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))  # 1D-only
    rec = store.get(0).record
    assert set(rec.modes_1d) == {"q_total", "q_ip"}    # 1D accumulated
    assert rec.modes_2d == ("qip_qoop",)               # 2D PRESERVED (not dropped)


def test_end_reintegrate_drops_unconsumed_carryover():
    # P1 (codex): a stopped/skipped reintegrate must not leave stale carry-over
    # that a later rehydration/upsert would merge.
    store = PublicationStore()
    store.upsert(_mode_pub(0, mode_1d="q_total", generation=store.generation))
    store.upsert(_mode_pub(1, mode_1d="q_total", generation=store.generation))
    store.begin_reintegrate()                       # carries 0 and 1
    store.upsert(_mode_pub(0, mode_1d="q_ip", generation=store.generation))  # only 0 republished
    store.end_reintegrate()                         # pass ended; frame 1 was skipped
    store.upsert(_mode_pub(1, mode_1d="q_oop", generation=store.generation))  # later (re)hydrate
    assert store.get(1).record.modes_1d == ("q_oop",)            # no stale q_total merged
    assert set(store.get(0).record.modes_1d) == {"q_total", "q_ip"}  # 0 accumulated normally


def test_same_source_id_suffix_match_rejects_different_dir():
    # P2 (codex): suffix-match by path components, not bare basename — abs/rel of
    # the SAME file merges; two different directories sharing a filename do NOT.
    from xdart.modules.frame_publication import _same_source_id
    assert _same_source_id("/data/run1/frame_0001.tif", "frame_0001.tif")        # abs vs bare rel
    assert _same_source_id("/data/run1/frame_0001.tif", "run1/frame_0001.tif")   # abs vs rel+dir
    assert not _same_source_id("run1/frame_0001.tif", "run2/frame_0001.tif")     # different dirs
    assert not _same_source_id("/data/run1/frame_0001.tif", "/data/run2/frame_0001.tif")
    assert not _same_source_id("", "frame_0001.tif")                             # known beats unknown
    assert _same_source_id("", "")                                               # transition unknown+unknown


def test_tier0_eviction_honors_persist_gate_mem1_15():
    """MEM1-15: with the evictability probe wired (the live widget wires it to
    the active FrameRecordStore), tier-0 eviction drops only labels whose
    records are PERSISTED; an unsaved ("owed") publication pins memory instead
    of being dropped — dropping it would blank the frame on scroll-back
    because this store is the cake's only render source.  Real record store
    on the probe side, real publications throughout."""
    from xrd_tools.session.frame_record_store import FrameRecordStore

    records = FrameRecordStore(max_items=None, max_heavy_items=None)
    store = PublicationStore(max_items=2, max_heavy_items=None)

    def evictable(label):
        if records.get(label) is None:
            return True
        return records.is_persisted(label)

    store.set_evictable_probe(evictable)

    for idx in (1, 2, 3):
        pub = publication_from_live_frame(DuckFrame(idx=idx))
        assert pub.record is not None
        records.upsert(pub.record)
        store.upsert(pub)

    # Nothing persisted yet: every label is owed -> the store EXCEEDS its cap
    # rather than dropping an unsaved frame.
    assert store.labels() == (1, 2, 3)

    # The writer's flush marks 1 persisted -> the next enforce evicts label 1
    # (oldest evictable) and ONLY label 1.
    records.mark_persisted(1)
    pub4 = publication_from_live_frame(DuckFrame(idx=4))
    records.upsert(pub4.record)
    store.upsert(pub4)
    assert store.labels() == (2, 3, 4)   # 1 evicted; 2,3 still owed -> pinned

    # Once everything persists, the bound is enforced normally again.
    records.mark_persisted([2, 3, 4])
    pub5 = publication_from_live_frame(DuckFrame(idx=5))
    records.upsert(pub5.record)
    store.upsert(pub5)
    assert store.labels() == (4, 5)

    # A label the record store never saw is NOT an owed live frame: it stays
    # evictable (no permanent pinning for untracked labels).
    store.upsert(publication_from_live_frame(DuckFrame(idx=99)))
    assert 99 in store.labels() and len(store.labels()) == 2


def test_tier0_eviction_without_probe_keeps_legacy_behavior():
    """No probe registered (viewer/browse stores) -> unchanged oldest-drop."""
    store = PublicationStore(max_items=2, max_heavy_items=None)
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))
    assert store.labels() == (2, 3)


# --------------------------------------------------------------------------- #
# X1 Slice 3c (S3-OR1): explicit immutable publication scan ownership
# --------------------------------------------------------------------------- #

def test_publication_scan_owner_stamped_and_default_none():
    stamped = publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a")
    assert stamped.scan_key == "run_a"
    unstamped = publication_from_live_frame(DuckFrame(idx=2))
    assert unstamped.scan_key is None


def test_scan_owner_preserved_through_thinning_tiers():
    """Tier-1 (semilight) and tier-2 (lightweight) eviction keep the owner."""
    store = PublicationStore(
        max_items=None, max_heavy_items=0, max_thumbnail_items=0)
    store.upsert(publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a"))
    thinned = store.get(1)
    assert thinned is not None
    assert thinned.raw_status == "evicted"          # fully thinned (both tiers)
    assert thinned.scan_key == "run_a"              # owner survives eviction


def test_scan_owner_preserved_through_same_scan_merge():
    """The upsert merge keeps the owner — including when a legacy UNSTAMPED
    republish of the same source merges onto a stamped entry."""
    store = PublicationStore(max_items=None, max_heavy_items=None)
    store.upsert(publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a"))
    # stamped incoming, same source: merged record + owner kept
    merged = store.upsert(
        publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a"))
    assert merged.scan_key == "run_a"
    # legacy unstamped incoming, same source: EXISTING owner is preserved
    merged = store.upsert(publication_from_live_frame(DuckFrame(idx=1)))
    assert merged.scan_key == "run_a"


def test_scan_owner_mismatch_prevents_same_source_record_merge():
    """A shared source path cannot splice modes across explicit scan owners."""
    store = PublicationStore(max_items=None, max_heavy_items=None)
    store.upsert(_mode_pub(
        1, mode_1d="q_total", source_identity="shared.nxs",
        scan_key="run_a"))
    replaced = store.upsert(_mode_pub(
        1, mode_1d="q_ip", source_identity="shared.nxs",
        scan_key="run_b"))

    assert replaced.scan_key == "run_b"
    assert set(replaced.record.modes_1d) == {"q_ip"}


def test_scan_owner_merge_normalizes_windows_spelling():
    store = PublicationStore(max_items=None, max_heavy_items=None)
    store.upsert(_mode_pub(
        1, mode_1d="q_total", source_identity="shared.nxs",
        scan_key=r"C:\Data\nested\..\run_a.nxs"))
    merged = store.upsert(_mode_pub(
        1, mode_1d="q_ip", source_identity="shared.nxs",
        scan_key="c:/data/run_a.nxs"))

    assert set(merged.record.modes_1d) == {"q_total", "q_ip"}


def test_scan_owner_preserved_through_reintegrate_carryover():
    store = PublicationStore(max_items=None, max_heavy_items=None)
    store.upsert(publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a"))
    store.begin_reintegrate()
    # the reintegrate republish (same source) restores the carried owner even
    # from a legacy unstamped caller
    republished = store.upsert(publication_from_live_frame(DuckFrame(idx=1)))
    assert republished.scan_key == "run_a"
    store.end_reintegrate()


def test_scan_owner_mismatch_prevents_reintegrate_carryover_merge():
    store = PublicationStore(max_items=None, max_heavy_items=None)
    store.upsert(_mode_pub(
        1, mode_1d="q_total", source_identity="shared.nxs",
        scan_key="run_a"))
    store.begin_reintegrate()
    republished = store.upsert(_mode_pub(
        1, mode_1d="q_ip", source_identity="shared.nxs",
        scan_key="run_b"))

    assert republished.scan_key == "run_b"
    assert set(republished.record.modes_1d) == {"q_ip"}
    store.end_reintegrate()


def test_scan_owner_preserved_through_hydration_replacement():
    """get_or_hydrate replacing an evicted payload keeps the owner even when
    the hydrator returns an unstamped same-source publication."""
    store = PublicationStore(
        max_items=None, max_heavy_items=0, max_thumbnail_items=0)
    store.upsert(publication_from_live_frame(DuckFrame(idx=1), scan_key="run_a"))
    assert store.get(1).raw_status == "evicted"
    store.set_hydrator(
        lambda label: publication_from_live_frame(
            DuckFrame(idx=1), generation=store.generation))
    hydrated = store.get_or_hydrate(1)
    assert hydrated is not None
    assert hydrated.view.intensity_1d is not None    # payload restored
    assert hydrated.scan_key == "run_a"              # owner survived


def test_gui_light_1d_completeness_query_does_not_compose_publications(
    monkeypatch,
):
    from xdart.gui.tabs.scattering.display_runtime import (
        publication_needs_hydration,
    )

    store, _lease, _allocation, _authority, build = _bound_gui_light_graph()
    store.publish_gui_light_1d(*build(0))
    store.publish_gui_light_1d(*build(1, heavy=False))
    store.upsert(FramePublication(
        FrameView(label=2),
        generation=store.generation,
        source_identity="scan.nxs#empty",
    ))
    labels = (0, 1, 2, 99)
    expected = frozenset(
        label for label in labels
        if not publication_needs_hydration(store.get(label), None)
    )
    assert expected == frozenset({0, 1})

    def fail_compose(*_args, **_kwargs):
        raise AssertionError("completeness query materialized a publication")

    monkeypatch.setattr(store, "_compose_locked", fail_compose)
    assert store.complete_labels(labels) == expected


def test_gui_light_1d_pair_is_frame_equivalent_with_one_zero_copy_owner():
    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    publication, light_record = build(0)

    composed = store.publish_gui_light_1d(publication, light_record)
    assert type(composed) is FramePublication
    assert type(store._items[0]) is FramePublication
    assert not store._items[0].view.has_1d
    assert all(
        mode_view.axis_1d is None
        and mode_view.intensity_1d is None
        and mode_view.sigma_1d is None
        for mode_view in store._items[0].record.results_1d.values()
    )
    assert_frameview_equivalent(composed.view, publication.view)
    assert_framerecord_equivalent(composed.record, publication.record)

    canonical = lease.borrow(0)
    assert canonical is not None
    try:
        assert composed.view.axis_1d.values is canonical.modes["bg"].coordinate
        assert composed.view.intensity_1d is canonical.modes["bg"].intensity
        assert composed.view.sigma_1d is canonical.modes["bg"].uncertainty
        assert (composed.record.results_1d["raw"].axis_1d.values
                is canonical.modes["raw"].coordinate)
        assert (composed.record.results_1d["raw"].intensity_1d
                is canonical.modes["raw"].intensity)
        assert (composed.record.results_1d["raw"].sigma_1d
                is canonical.modes["raw"].uncertainty)
    finally:
        canonical.close()
    census = store.ndarray_owner_census()
    expected_heavy = frozenset(
        id(store._ndarray_root(value))
        for value in (
            store._items[0].view.intensity_2d,
            store._items[0].view.raw,
            store._items[0].view.thumbnail,
        )
    )
    assert expected_heavy <= census["publication"]
    assert census["publication"].isdisjoint(census["lease"])


def test_gui_light_1d_publication_uses_constant_time_lease_residency_queries(
    monkeypatch,
):
    from xrd_tools.session import Light1DRetentionLease

    store, lease, _allocation, _authority, build = _bound_gui_light_graph(
        rows=3,
    )
    for label in (0, 1, 2):
        store.publish_gui_light_1d(*build(label))

    def forbidden_prefix_snapshot(owner):
        if owner is lease:
            raise AssertionError("GUI publication materialized the lease prefix")
        return original_keys(owner)

    original_keys = Light1DRetentionLease.keys
    monkeypatch.setattr(
        Light1DRetentionLease, "keys", forbidden_prefix_snapshot,
    )
    composed = store.publish_gui_light_1d(*build(3))

    assert composed.label == 3 and composed.view.has_1d
    assert store.labels() == (1, 2, 3)
    assert lease.retained_count == 3
    assert lease.oldest_row_identity == 1
    assert not lease.contains(0) and lease.contains(3)


def test_light_1d_publication_uses_constant_time_lease_residency_queries(
    monkeypatch,
):
    from xrd_tools.session import Light1DRetentionLease

    store, lease, _allocation, _authority, build = _bound_gui_light_graph(
        rows=3,
    )
    for label in (0, 1, 2):
        publication, light_record = build(label)
        store.publish_light_1d(
            light_record, source_identity=publication.source_identity,
        )

    original_keys = Light1DRetentionLease.keys

    def forbidden_prefix_snapshot(owner):
        if owner is lease:
            raise AssertionError("publication materialized the lease prefix")
        return original_keys(owner)

    monkeypatch.setattr(
        Light1DRetentionLease, "keys", forbidden_prefix_snapshot,
    )
    publication, light_record = build(3)
    shell = store.publish_light_1d(
        light_record, source_identity=publication.source_identity,
    )

    assert shell.label == 3
    assert store.get_light_1d_shell(0) is None
    assert store.get_light_1d_shell(3) is shell
    assert lease.retained_count == 3
    assert lease.oldest_row_identity == 1


def test_gui_light_1d_pair_refuses_identity_authority_mismatches_atomically():
    from xrd_tools.session import Light1DStaleGeneration

    store, lease, allocation, _authority, build = _bound_gui_light_graph()
    publication, light_record = build(0)
    store.publish_gui_light_1d(publication, light_record)
    prior_base = store._items[0]
    prior_pair = store._light_1d_items[0]
    prior_guard = prior_pair.guard
    prior_keys = lease.keys()
    prior_generation = store.generation
    prior_labels = store.labels()
    prior_heavy = tuple(store._heavy_labels)
    prior_thumbs = tuple(store._thumb_labels)

    candidates = []
    publication_1, light_1 = build(1)
    candidates.append((publication_1, replace(light_1, row_identity=99)))
    candidates.append(build(1, publication_source="foreign-source"))
    candidates.append(build(1, publication_scan="foreign-scan"))
    candidates.append(build(1, publication_generation=store.generation + 1))
    candidates.append(build(1, light_generation=lease.generation + 1))
    for candidate_publication, candidate_record in candidates:
        with pytest.raises((ValueError, Light1DStaleGeneration)):
            store.publish_gui_light_1d(candidate_publication, candidate_record)
        assert store._items[0] is prior_base
        assert store._light_1d_items[0] is prior_pair
        assert prior_pair.guard is prior_guard and not prior_guard.closed
        assert lease.keys() == prior_keys
        assert store.generation == prior_generation

    store.allocation = object()
    try:
        with pytest.raises(ValueError, match="allocation"):
            store.publish_gui_light_1d(*build(1))
        assert store._items[0] is prior_base
        assert store._light_1d_items[0] is prior_pair
        assert lease.keys() == prior_keys
        assert store.generation == prior_generation
    finally:
        store.allocation = allocation

    failures = []
    for same_owner in (False, True):
        for candidate_label in (0, 1):
            graph = _bound_gui_light_graph(
                rows=1, same_owner_coordinate=same_owner,
            )
            case_store, case_lease, _a, case_authority, case_build = graph
            case_store.publish_gui_light_1d(*case_build(0))
            case_base = case_store._items[0]
            case_pair = case_store._light_1d_items[0]
            case_guard = case_pair.guard
            before = (
                case_store.labels(), case_lease.keys(), case_store.generation,
                tuple(case_store._heavy_labels), tuple(case_store._thumb_labels),
                case_lease.owned_buffer_ids,
                case_lease.unique_owned_ndarray_bytes,
                case_authority.snapshot(),
            )
            bad, bad_light = case_build(
                candidate_label, reverse_bg_coordinate=same_owner,
            )
            if not same_owner:
                shared = np.arange(4.0)
                raw_view = replace(
                    bad.record.results_1d["raw"],
                    axis_1d=replace(
                        bad.record.results_1d["raw"].axis_1d, values=shared,
                    ),
                    intensity_1d=shared,
                )
                bad = replace(bad, record=replace(
                    bad.record,
                    results_1d={**bad.record.results_1d, "raw": raw_view},
                ))
                bad_light = replace(bad_light, modes={
                    **bad_light.modes,
                    "raw": replace(
                        bad_light.modes["raw"],
                        coordinate=shared, intensity=shared,
                    ),
                })
            try:
                case_store.publish_gui_light_1d(bad, bad_light)
            except ValueError:
                pass
            else:
                failures.append((same_owner, candidate_label, "no refusal"))
            after = (
                case_store.labels(), case_lease.keys(), case_store.generation,
                tuple(case_store._heavy_labels), tuple(case_store._thumb_labels),
                case_lease.owned_buffer_ids,
                case_lease.unique_owned_ndarray_bytes,
                case_authority.snapshot(),
            )
            if not (
                after == before
                and case_store._items.get(0) is case_base
                and case_store._light_1d_items.get(0) is case_pair
                and case_pair.guard is case_guard
                and not case_guard.closed
            ):
                failures.append((same_owner, candidate_label, "state changed"))
    assert not failures, failures


def test_gui_light_1d_pair_refuses_unsupported_layouts_atomically():
    graph_cases = (
        _bound_gui_light_graph(dtype=np.float32),
        _bound_gui_light_graph(dtype=np.dtype(">f8")),
        _bound_gui_light_graph(shared_coordinate=True),
    )
    for store, lease, _allocation, _authority, build in graph_cases:
        with pytest.raises(ValueError):
            store.publish_gui_light_1d(*build(0))
        assert store._items == {}
        assert store._light_1d_items == {}
        assert lease.keys() == ()

    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    invalid = (
        build(0, array_length=3),
        build(0, drop_light_mode="raw"),
        build(0, light_active="raw"),
        build(0, omit_raw_uncertainty=True),
        build(0, add_bg_uncertainty=True),
        build(0, raw_ref=object()),
    )
    for publication, light_record in invalid:
        with pytest.raises(ValueError):
            store.publish_gui_light_1d(publication, light_record)
        assert store._items == {}
        assert store._light_1d_items == {}
        assert lease.keys() == ()


def test_bound_upsert_refuses_1d_and_allows_raw_2d_only():
    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    publication, _light_record = build(0)

    with pytest.raises(ValueError, match="1-D"):
        store.upsert(publication)
    assert store._items == {}
    assert lease.keys() == ()

    array_free = _without_1d(publication)
    hidden_raw = replace(array_free, raw_ref=DuckFrame(idx=0))
    with pytest.raises(ValueError, match="raw_ref"):
        store.upsert(hidden_raw)
    assert store._items == {}
    assert store._light_1d_items == {}
    assert lease.keys() == ()

    stored = store.upsert(array_free)
    assert stored is store._items[0]
    assert not stored.view.has_1d
    assert stored.view.has_2d
    assert stored.view.raw is publication.view.raw


def test_bound_refresh_preserves_qualified_pair_and_retires_foreign_pair():
    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    publication, light_record = build(0)
    store.publish_gui_light_1d(publication, light_record)
    pair = store._light_1d_items[0]
    canonical = lease.borrow(0)
    assert canonical is not None
    canonical_intensity = canonical.modes["bg"].intensity
    canonical.close()

    refresh, _unused = build(0)
    refreshed = store.upsert(_without_1d(refresh))
    assert store._light_1d_items[0] is pair
    assert lease.keys() == (0,)
    assert refreshed.view.intensity_1d is canonical_intensity
    assert refreshed.view.has_2d

    for mismatch in ("source", "scan", "generation"):
        other_store, other_lease, _a, _r, other_build = _bound_gui_light_graph()
        original, original_light = other_build(0)
        other_store.publish_gui_light_1d(original, original_light)
        options = {
            "source": {"publication_source": "foreign-source"},
            "scan": {"publication_scan": "foreign-scan"},
            "generation": {"publication_generation": other_store.generation + 1},
        }[mismatch]
        foreign, _unused = other_build(0, **options)
        replaced_publication = other_store.upsert(_without_1d(foreign))
        assert not replaced_publication.view.has_1d
        assert 0 not in other_store._light_1d_items
        assert other_lease.keys() == ()


def test_bound_reads_compose_frame_publications_without_shell_leakage():
    from xdart.modules.frame_publication import Light1DPublicationShell

    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    publication, light_record = build(0)
    shell = store.publish_gui_light_1d(publication, light_record)
    assert type(shell) is FramePublication

    canonical = lease.borrow(0)
    assert canonical is not None
    try:
        values = (
            store.get(0),
            store.get_many((0,))[0],
            store.snapshot()[0],
        )
        for value in values:
            assert type(value) is FramePublication
            assert not isinstance(value, Light1DPublicationShell)
            assert value.view.intensity_1d is canonical.modes["bg"].intensity
            assert (value.record.results_1d["raw"].axis_1d.values
                    is canonical.modes["raw"].coordinate)
    finally:
        canonical.close()
    public_shell = store.get_light_1d_shell(0)
    assert public_shell is not None
    assert not hasattr(public_shell, "borrow")


def test_bound_heavy_and_thumbnail_demotion_preserve_light_pair():
    store, lease, _allocation, _authority, build = _bound_gui_light_graph()
    publication, light_record = build(0)
    store.publish_gui_light_1d(publication, light_record)
    pair = store._light_1d_items[0]

    assert store.evict_heavy(0)
    semilight = store.get(0)
    assert store._light_1d_items[0] is pair
    assert lease.keys() == (0,)
    assert semilight.view.has_1d
    assert semilight.view.intensity_2d is None
    assert semilight.view.thumbnail is not None

    assert store.evict_thumbnail(0)
    light = store.get(0)
    assert store._light_1d_items[0] is pair
    assert lease.keys() == (0,)
    assert light.view.has_1d
    assert light.view.thumbnail is None


def test_bound_terminal_removals_close_exact_light_pairs(monkeypatch):
    from xrd_tools.session import (
        Light1DBorrow,
        Light1DCleanupHooks,
        Light1DCleanupPending,
        Light1DCleanupToken,
        Light1DLeaseState,
        Light1DRetentionLease,
        Light1DStaleGeneration,
        Light1DUnavailable,
    )

    def published(*, rows=3, labels=(0,)):
        graph = _bound_gui_light_graph(rows=rows)
        local_store, local_lease, _allocation, _authority, build = graph
        for label in labels:
            local_store.publish_gui_light_1d(*build(label))
        return graph

    store, lease, _a, _r, _build = published(labels=(0, 1))
    store.invalidate((0,))
    assert lease.keys() == (1,)
    assert 0 not in store._light_1d_items and store.get(0) is None
    assert store.discard(1)
    assert lease.keys() == ()

    store, lease, _a, _r, build = published(rows=1)
    store.publish_gui_light_1d(*build(1))
    assert lease.keys() == (1,)
    assert 0 not in store._light_1d_items
    assert store.get(0) is None

    store, lease, _a, _r, _build = published(labels=(0,))
    client = lease.borrow(0)
    prior_base = store._items[0]
    with pytest.raises(Light1DUnavailable):
        store.discard(0)
    assert store._items[0] is prior_base
    assert 0 in store._light_1d_items
    assert lease.keys() == (0,)
    assert client is not None
    client.close()
    assert store.discard(0)

    store, lease, allocation, _r, _build = published(labels=(0, 1, 2))
    client = lease.borrow(1)
    with pytest.raises(Light1DUnavailable):
        store.clear()
    assert lease.keys() == (1, 2)
    assert store.get(0) is None
    assert store.get(1) is not None and store.get(2) is not None
    assert store.allocation is allocation and store._light_1d is lease
    assert client is not None
    client.close()
    store.clear()
    assert lease.keys() == ()
    assert store._items == {} and store._light_1d_items == {}
    assert store.allocation is allocation and store._light_1d is lease

    for state in ("fenced", "cleanup-pending"):
        for operation in ("invalidate", "discard", "total-bound", "foreign"):
            store, lease, _a, _r, build = published(labels=(0,))
            if operation == "total-bound":
                store._max_items = 0
            if state == "fenced":
                lease.fence()
            else:
                def fail_cancel():
                    raise RuntimeError("injected cancel refusal")

                hooks = store.light_1d_cleanup_hooks(
                    lease, cancel=fail_cancel,
                )
                with pytest.raises(Light1DCleanupPending):
                    lease.release(reason="pending-before-clear", hooks=hooks)
            base = store._items[0]
            pair = store._light_1d_items[0]
            guard = pair.guard
            before = (
                store.labels(), lease.keys(), store.generation,
                tuple(store._heavy_labels), tuple(store._thumb_labels),
            )
            actions = {
                "invalidate": lambda: store.invalidate((0,)),
                "discard": lambda: store.discard(0),
                "total-bound": store._enforce_bounds_locked,
                "foreign": lambda: store.upsert(_without_1d(
                    build(0, publication_source="foreign-source")[0]
                )),
            }
            with pytest.raises((
                RuntimeError, Light1DStaleGeneration, Light1DUnavailable,
            )):
                actions[operation]()
            assert store._items[0] is base
            assert store._light_1d_items[0] is pair
            assert pair.guard is guard and not guard.closed
            assert (
                store.labels(), lease.keys(), store.generation,
                tuple(store._heavy_labels), tuple(store._thumb_labels),
            ) == before

    store, lease, allocation, _r, _build = published(labels=(0,))
    hooks = store.light_1d_cleanup_hooks(lease)
    before = (store._items.copy(), store._light_1d_items.copy(), lease.keys())
    with pytest.raises(RuntimeError):
        hooks.clear()
    assert (store._items, store._light_1d_items, lease.keys()) == before
    failures = []
    thread = threading.Thread(
        target=lambda: failures.append(pytest.raises(RuntimeError, hooks.clear)),
    )
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and len(failures) == 1

    # Each private callback authority fact is independently observable while
    # the other facts are qualified.
    store, lease, _a, _r, _build = published(labels=(0,))
    frozen = store.light_1d_cleanup_hooks(lease)
    foreign = store.light_1d_cleanup_hooks(lease)
    rogue = Light1DCleanupHooks(
        clear=foreign.clear,
        detach=foreign.detach,
    )
    with pytest.raises(Light1DCleanupPending) as pending:
        lease.release(reason="foreign-hooks", hooks=rogue)
    assert pending.value.receipt.failed_step == "clear"
    assert lease.keys() == (0,) and 0 in store._light_1d_items

    store, lease, _a, _r, _build = published(labels=(0,))
    hooks = None

    def erase_cleanup_token():
        lease._cleanup_token = None

    hooks = store.light_1d_cleanup_hooks(lease, drain=erase_cleanup_token)
    with pytest.raises(Light1DCleanupPending) as pending:
        lease.release(reason="missing-token", hooks=hooks)
    assert pending.value.receipt.failed_step == "clear"
    assert lease.keys() == (0,) and 0 in store._light_1d_items

    store, lease, _a, _r, _build = published(labels=(0,))
    hooks = store.light_1d_cleanup_hooks(lease)
    with lease._lock:
        lease._cleanup_hooks = hooks
        lease._cleanup_token = Light1DCleanupToken(
            lease.grant_id, lease.generation, 1,
        )
        lease._cleanup_step = 2
        lease._state = Light1DLeaseState.FENCED
        lease._cleanup_callback_call = (
            threading.get_ident(), "clear", hooks.clear,
        )
    cross_thread_failures = []
    thread = threading.Thread(target=lambda: cross_thread_failures.append(
        pytest.raises(RuntimeError, hooks.clear),
    ))
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and len(cross_thread_failures) == 1
    assert lease.keys() == (0,) and 0 in store._light_1d_items
    with lease._lock:
        lease._cleanup_step = 1
    with pytest.raises(RuntimeError):
        hooks.clear()
    assert lease.keys() == (0,) and 0 in store._light_1d_items
    with lease._lock:
        lease._cleanup_step = 2
        lease._state = Light1DLeaseState.ACTIVE
    with pytest.raises(RuntimeError):
        hooks.clear()
    assert lease.keys() == (0,) and 0 in store._light_1d_items

    retire_calls = []
    original_retire = Light1DRetentionLease.retire

    def record_retire(owner, *args, **kwargs):
        retire_calls.append((owner, args, kwargs))
        return original_retire(owner, *args, **kwargs)

    monkeypatch.setattr(Light1DRetentionLease, "retire", record_retire)
    receipt = lease.release(reason="terminal", hooks=hooks)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert retire_calls == []
    assert lease.state is Light1DLeaseState.RELEASED
    assert store.allocation is None and store._light_1d is None
    assert store._items == {} and store._light_1d_items == {}
    with pytest.raises(RuntimeError):
        hooks.detach()

    # A cursor-2 guard-close failure keeps that pair for the exact retry; the
    # lease-owned canonical clear and cursor-3 detach then run once.
    store, lease, _a, _r, _build = published(labels=(0, 1))
    hooks = store.light_1d_cleanup_hooks(lease)
    target_guard = store._light_1d_items[1].guard
    original_close = Light1DBorrow.close
    fail_once = [True]

    def flaky_close(owner):
        if owner is target_guard and fail_once:
            fail_once.pop()
            raise RuntimeError("injected guard close")
        return original_close(owner)

    monkeypatch.setattr(Light1DBorrow, "close", flaky_close)
    with pytest.raises(Light1DCleanupPending) as pending:
        lease.release(reason="terminal", hooks=hooks)
    assert pending.value.receipt.failed_step == "clear"
    assert 0 not in store._light_1d_items
    assert 1 in store._light_1d_items
    assert lease.keys() == (0, 1)
    monkeypatch.setattr(Light1DBorrow, "close", original_close)
    lease.retry_cleanup(pending.value.token, hooks=hooks)
    assert lease.state is Light1DLeaseState.RELEASED
    assert store.allocation is None and store._light_1d is None

    # A hidden 1-D base is a pre-mutation cursor-3 refusal. Qualified raw/2-D
    # bases are admitted and consumed by the retry.
    store, lease, allocation, _r, build = _bound_gui_light_graph()
    hidden, _hidden_light = build(0)
    store._items[0] = hidden
    hooks = store.light_1d_cleanup_hooks(lease)
    with pytest.raises(Light1DCleanupPending) as pending:
        lease.release(reason="terminal", hooks=hooks)
    assert pending.value.receipt.failed_step == "detach"
    assert store._items[0] is hidden
    assert store.allocation is allocation and store._light_1d is lease
    hidden_raw = replace(_without_1d(hidden), raw_ref=DuckFrame(idx=0))
    store._items[0] = hidden_raw
    with pytest.raises(Light1DCleanupPending) as raw_pending:
        lease.retry_cleanup(pending.value.token, hooks=hooks)
    assert raw_pending.value.receipt.failed_step == "detach"
    assert store._items[0] is hidden_raw
    assert store.allocation is allocation and store._light_1d is lease
    store._items[0] = _without_1d(hidden)
    lease.retry_cleanup(raw_pending.value.token, hooks=hooks)
    assert store._items == {}
    assert store.allocation is None and store._light_1d is None

    # A pair whose public retirement falsely reports an absent key is never
    # allowed to diverge from the lease map or public base.
    store, lease, _a, _r, _build = published(labels=(0,))
    prior_base = store._items[0]
    prior_retire = Light1DRetentionLease.retire
    monkeypatch.setattr(Light1DRetentionLease, "retire", lambda *_a, **_k: False)
    with pytest.raises(RuntimeError, match="inconsistent"):
        store.discard(0)
    assert store._items[0] is prior_base
    assert 0 in store._light_1d_items and lease.keys() == (0,) and not store._light_1d_items[0].guard.closed
    monkeypatch.setattr(Light1DRetentionLease, "retire", prior_retire)


def test_gui_light_1d_escaped_alias_reconciles_without_regrant_then_retries(
    monkeypatch,
):
    from xrd_tools.session import Light1DRetentionLease, Light1DUnavailable
    import xrd_tools.session.light_1d_retention as retention_module

    failures = []

    def caught(call):
        try:
            return None, call()
        except BaseException as exc:
            return exc, None

    def state(store, lease):
        return (
            store.labels(), tuple(map(id, store._items.values())),
            tuple((label, id(pair), id(pair.guard), pair.guard.closed)
                  for label, pair in store._light_1d_items.items()),
            lease.keys(), store.generation, tuple(store._heavy_labels),
            tuple(store._thumb_labels), lease.owned_buffer_ids,
            lease.unique_owned_ndarray_bytes,
        )

    def recording_probe(calls, accepted):
        def probe(label):
            calls.append(label)
            return label in accepted
        return probe

    def tracked(calls, name, method):
        def wrapper(owner, *args, **kwargs):
            calls.append(name)
            return method(owner, *args, **kwargs)
        return wrapper

    store, lease, _allocation, _authority, build = _bound_gui_light_graph(rows=1)
    store.publish_gui_light_1d(*build(0))
    shown = store.get(0)
    alias = shown.view.intensity_1d
    owned_before = lease.unique_owned_ndarray_bytes
    calls = []
    with monkeypatch.context() as observed:
        observed.setattr(
            Light1DRetentionLease, "retain",
            tracked(calls, "retain", Light1DRetentionLease.retain),
        )
        observed.setattr(
            Light1DRetentionLease, "_canonicalize_record",
            tracked(calls, "canonicalize", Light1DRetentionLease._canonicalize_record),
        )
        observed.setattr(
            Light1DRetentionLease, "_private_array_copy",
            tracked(calls, "copy", Light1DRetentionLease._private_array_copy),
        )
        missed = store.publish_gui_light_1d(*build(1))
        before_retry = tuple(calls), lease.unique_owned_ndarray_bytes
        del alias, shown
        gc.collect()
        retried = store.publish_gui_light_1d(*build(1))
    if not (
        not missed.view.has_1d
        and store.get(0) is None
        and 0 not in store._light_1d_items
        and before_retry == ((), owned_before)
        and calls.count("retain") == 1
        and retried.view.has_1d
        and lease.keys() == (1,)
    ):
        failures.append(("retired receipt admission", before_retry, calls))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=2)
    probe_calls = []
    store.set_evictable_probe(recording_probe(probe_calls, set()))
    store.upsert(_without_1d(build(2)[0]))
    store.publish_gui_light_1d(*build(0))
    store.upsert(_without_1d(build(3)[0]))
    client = lease.borrow(0)
    before = state(store, lease)
    probe_calls.clear()
    store.set_evictable_probe(recording_probe(probe_calls, {2, 0}))
    error, _ = caught(lambda: store.publish_gui_light_1d(*build(1)))
    if not (
        isinstance(error, Light1DUnavailable)
        and state(store, lease) == before
        and probe_calls == [2, 0]
        and all(probe_calls.count(label) == 1 for label in set(probe_calls))
    ):
        failures.append(("unpaired then borrowed victim", probe_calls))
    client.close()

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=3)
    store.set_evictable_probe(lambda _label: False)
    store.publish_gui_light_1d(*build(0))
    store.publish_gui_light_1d(*build(1))
    store.upsert(_without_1d(build(2)[0]))
    store.upsert(_without_1d(build(3)[0]))
    client = lease.borrow(1)
    before = state(store, lease)
    probe_calls = []
    store.set_evictable_probe(recording_probe(probe_calls, {0, 1}))
    error, _ = caught(lambda: store.publish_gui_light_1d(*build(4)))
    if not (
        isinstance(error, Light1DUnavailable)
        and state(store, lease) == before
        and probe_calls == [0, 1]
    ):
        failures.append(("multiple paired victims", probe_calls))
    client.close()

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=3)
    store.set_evictable_probe(lambda _label: False)
    store.publish_gui_light_1d(*build(0))
    store.upsert(_without_1d(build(2)[0]))
    store.upsert(_without_1d(build(3)[0]))
    before = state(store, lease)
    probe_calls = []
    store.set_evictable_probe(recording_probe(probe_calls, {1}))
    error, result = caught(lambda: store.publish_gui_light_1d(*build(1)))
    if not (
        isinstance(error, Light1DUnavailable)
        and result is None and state(store, lease) == before
        and probe_calls == [0, 2, 3, 1]
        and 1 not in store._items and 1 not in store._light_1d_items
    ):
        failures.append(("incoming selected", probe_calls, error, result))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=2)
    store.publish_gui_light_1d(*build(0))
    store.publish_gui_light_1d(*build(1))
    preserved = store._light_1d_items[0]
    probe_calls, preflights, retirements = [], [], []
    original_preflight = Light1DRetentionLease._preflight_store_record
    original_retire = Light1DRetentionLease.retire

    def record_preflight(owner, *args, **kwargs):
        preflights.append(tuple(row for row, _guard in kwargs.get("retiring", ())))
        return original_preflight(owner, *args, **kwargs)

    def record_retire(owner, row, **kwargs):
        retirements.append(row)
        return original_retire(owner, row, **kwargs)

    store.set_evictable_probe(recording_probe(probe_calls, {1}))
    with monkeypatch.context() as total:
        total.setattr(Light1DRetentionLease, "_preflight_store_record", record_preflight)
        total.setattr(Light1DRetentionLease, "retire", record_retire)
        result = store.publish_gui_light_1d(*build(2))
    if not (
        result.view.has_1d and store.labels() == (0, 2)
        and store._light_1d_items[0] is preserved
        and probe_calls == [0, 1]
        and sum(entry.count(1) for entry in preflights) == 1
        and retirements == [1]
    ):
        failures.append(("total victim frees slot", probe_calls, preflights, retirements))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=2)
    store.set_evictable_probe(lambda _label: False)
    store.publish_gui_light_1d(*build(0))
    store.upsert(_without_1d(build(2)[0]))
    store.upsert(_without_1d(build(3)[0]))
    before = state(store, lease)
    probe_calls = []
    store.set_evictable_probe(recording_probe(probe_calls, {0}))
    error, result = caught(lambda: store.publish_gui_light_1d(*build(0)))
    if not (
        isinstance(error, Light1DUnavailable) and result is None
        and state(store, lease) == before and probe_calls == [2, 3, 0]
    ):
        failures.append(("reused incoming selected", probe_calls, error, result))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=2)
    store.set_evictable_probe(lambda _label: False)
    store.publish_gui_light_1d(*build(0))
    store.publish_gui_light_1d(*build(2))
    store.upsert(_without_1d(build(3)[0]))
    pair = store._light_1d_items[0]
    probe_calls, calls, retirements = [], [], []
    store.set_evictable_probe(recording_probe(probe_calls, {2}))
    with monkeypatch.context() as reused:
        reused.setattr(Light1DRetentionLease, "retain",
                       tracked(calls, "retain", Light1DRetentionLease.retain))
        reused.setattr(Light1DRetentionLease, "_canonicalize_record",
                       tracked(calls, "canonicalize",
                               Light1DRetentionLease._canonicalize_record))
        reused.setattr(Light1DRetentionLease, "_private_array_copy",
                       tracked(calls, "copy", Light1DRetentionLease._private_array_copy))
        reused.setattr(Light1DRetentionLease, "retire", record_retire)
        result = store.publish_gui_light_1d(*build(0))
    if not (
        result.view.has_1d and probe_calls == [2] and calls == []
        and retirements == [2]
        and store.labels() == (3, 0) and store._light_1d_items[0] is pair
        and lease.keys() == (0,) and not pair.guard.closed
    ):
        failures.append(("reused total victim", probe_calls, calls, retirements, store.labels()))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=2)
    store.set_evictable_probe(lambda _label: False)
    store.upsert(_without_1d(build(2)[0]))
    store.publish_gui_light_1d(*build(0))
    store.publish_gui_light_1d(*build(1))
    store.upsert(_without_1d(build(3)[0]))
    preserved = store._light_1d_items[1]
    probe_calls, preflights, retirements = [], [], []
    store.set_evictable_probe(recording_probe(probe_calls, {2, 0, 3}))
    with monkeypatch.context() as deduplicated:
        deduplicated.setattr(
            Light1DRetentionLease, "_preflight_store_record", record_preflight,
        )
        deduplicated.setattr(Light1DRetentionLease, "retire", record_retire)
        result = store.publish_gui_light_1d(*build(4))
    if not (
        result.view.has_1d and store._light_1d_items[1] is preserved
        and probe_calls == [2, 0, 1, 3]
        and sum(entry.count(0) for entry in preflights) == 1
        and retirements.count(0) == 1
    ):
        failures.append(("deduplicated capacity victim", probe_calls,
                         preflights, retirements))

    store, lease, _a, _r, build = _bound_gui_light_graph(rows=1)
    store.publish_gui_light_1d(*build(0))
    markers = []
    fatal = MemoryError("injected GUI canonical freeze fault")
    original_publish = PublicationStore.publish_gui_light_1d
    original_preflight = Light1DRetentionLease._preflight_store_record
    original_freeze = retention_module._freeze_array

    def mark_publish(owner, *args, **kwargs):
        markers.append("store")
        return original_publish(owner, *args, **kwargs)

    def mark_preflight(owner, *args, **kwargs):
        commit = kwargs.get("commit")
        if commit is not None:
            def marked_commit(retain_candidate):
                def marked_retain_candidate():
                    markers.append("retain_candidate")
                    return retain_candidate()
                return commit(marked_retain_candidate)
            kwargs["commit"] = marked_commit
        return original_preflight(owner, *args, **kwargs)

    def fail_canonical_freeze(array):
        original_freeze(array)
        if type(array) is memoryview:
            markers.append("freeze")
            raise fatal

    with monkeypatch.context() as seam:
        seam.setattr(PublicationStore, "publish_gui_light_1d", mark_publish)
        seam.setattr(Light1DRetentionLease, "_preflight_store_record", mark_preflight)
        seam.setattr(Light1DRetentionLease, "retain",
                     tracked(markers, "retain", Light1DRetentionLease.retain))
        seam.setattr(Light1DRetentionLease, "_canonicalize_record",
                     tracked(markers, "canonicalize",
                             Light1DRetentionLease._canonicalize_record))
        seam.setattr(Light1DRetentionLease, "_private_array_copy",
                     tracked(markers, "copy",
                             Light1DRetentionLease._private_array_copy))
        seam.setattr(retention_module, "_freeze_array", fail_canonical_freeze)
        error, _ = caught(lambda: store.publish_gui_light_1d(*build(1)))
    expected = ("store", "retain_candidate", "retain", "canonicalize", "copy", "freeze")
    if error is not fatal or tuple(markers[:len(expected)]) != expected:
        failures.append(("GUI production seam", markers, error))
    hooks = store.light_1d_cleanup_hooks(lease)
    store.clear()
    lease.release(reason="terminal", hooks=hooks)
    assert not failures, failures


def test_bound_reintegrate_refuses_before_any_mutation():
    store, lease, allocation, _authority, build = _bound_gui_light_graph()
    store.publish_gui_light_1d(*build(0))
    prior = (
        store.generation,
        store._items.copy(),
        store._light_1d_items.copy(),
        store._carryover.copy(),
        lease.keys(),
        store.allocation,
        store._light_1d,
    )

    with pytest.raises(RuntimeError, match="reintegrat"):
        store.begin_reintegrate()
    assert (
        store.generation,
        store._items,
        store._light_1d_items,
        store._carryover,
        lease.keys(),
        store.allocation,
        store._light_1d,
    ) == prior


def test_b1_raw_only_eviction_preserves_every_nonraw_view_identity():
    raw_a = np.full((4, 4), 7.0); raw_b = np.full((4, 4), 8.0); base = _mode_pub(7, mode_1d="q_total", mode_2d="q_chi")
    active = replace(base.view, raw=raw_a, mask_baked=True); alternate = replace(active, raw=raw_b)
    one_d = {mode: replace(view, raw=raw_a) for mode, view in base.record.results_1d.items()}
    publication = replace(
        base,
        view=active,
        record=replace(
            base.record,
            results_1d=one_d,
            results_2d={"q_chi": active, "q_ip_q_oop": alternate},
        ),
        raw_ref=DuckFrame(idx=7),
        raw_status="ready",
        source_identity="scan.nxs#7",
        scan_key="scan-owner",
    )
    store = PublicationStore(max_heavy_items=None, max_thumbnail_items=None); store.upsert(publication); before = store.get(7)
    preserved = {
        name: getattr(before.view, name)
        for name in (
            "axis_1d", "intensity_1d", "axis_2d_x", "axis_2d_y",
            "intensity_2d", "thumbnail",
        )
    }
    assert store.has_raw(7) and store.evict_raw(7)
    evicted = store.get(7)
    assert not store.has_raw(7) and not store.evict_raw(7)
    assert evicted.raw_ref is None and evicted.raw_status == "thumbnail"
    assert evicted.source_identity == "scan.nxs#7" and evicted.scan_key == "scan-owner"
    assert evicted.view.mask_baked is True
    assert all(getattr(evicted.view, name) is value for name, value in preserved.items())
    assert evicted.view.metadata_raw == before.view.metadata_raw and evicted.view.extra == before.view.extra
    assert all(view.raw is None for view in evicted.record.results_1d.values())
    assert all(view.raw is None for view in evicted.record.results_2d.values())
    assert set(evicted.record.results_2d) == {"q_chi", "q_ip_q_oop"}


def test_b1_bound_raw_overlay_is_additive_protected_and_identity_qualified():
    store, lease, _allocation, _authority, build = _bound_gui_light_graph(
        rows=3, heavy_rows=1)
    publication, light = build(0)
    publication = replace(publication, view=replace(publication.view, raw=None),
        record=replace(publication.record, results_1d={mode: replace(view, raw=None) for mode, view in publication.record.results_1d.items()},
            results_2d={mode: replace(view, raw=None) for mode, view in publication.record.results_2d.items()}),
        raw_status="thumbnail")
    store.publish_gui_light_1d(publication, light)
    before = store._items[0]; pair = store._light_1d_items[0]
    orders = (tuple(store._items), tuple(store._heavy_labels), tuple(store._thumb_labels), tuple(store._light_1d_items), lease.keys())
    raw = np.full((4, 4), 11.0); raw.setflags(write=False)
    installed = store.install_raw(0, raw, mask_baked=True); fields = ("axis_2d_x", "axis_2d_y", "intensity_2d", "thumbnail", "metadata_raw", "extra")
    assert installed is not None and store._items[0].view.raw is raw
    assert all(getattr(store._items[0].view, name) is getattr(before.view, name) for name in fields)
    assert all(getattr(getattr(store._items[0].record, dimension)[mode], name) is getattr(view, name) for dimension in ("results_1d", "results_2d") for mode, view in getattr(before.record, dimension).items() for name in fields)
    assert all(view.raw is raw for view in (*store._items[0].record.results_1d.values(), *store._items[0].record.results_2d.values()))
    assert store._items[0].raw_status == "ready" and store._items[0].view.mask_baked is True
    assert orders == (tuple(store._items), tuple(store._heavy_labels), tuple(store._thumb_labels), tuple(store._light_1d_items), lease.keys())

    store.publish_gui_light_1d(publication, light, protected=(0,))
    assert store.has_raw(0) and store._light_1d_items[0] is pair
    for label in (1, 2): store.publish_gui_light_1d(*build(label), protected=(0,))
    assert store.has_raw(0) and store._items[0].view.raw is raw
    foreign, _light = build(0, publication_source="foreign")
    frozen = (store._items[0], store._light_1d_items[0], lease.keys())
    with pytest.raises(ValueError, match="protected raw publication identity"):
        store.upsert(_without_1d(foreign), protected=(0,))
    assert frozen == (store._items[0], store._light_1d_items[0], lease.keys())
