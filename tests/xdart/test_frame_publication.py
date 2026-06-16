from __future__ import annotations

import numpy as np
import pytest
import h5py

from xrd_tools.core import (
    IntegrationResult1D,
    IntegrationResult2D,
    TwoDKind,
    assert_frameview_equivalent,
)
from xrd_tools.io.nexus import write_integrated_stack

from xdart.modules.frame_publication import (
    PublicationStore,
    publication_error_details,
    publication_from_nexus_frame,
    publication_from_live_frame,
    publication_has_1d_errors,
    publication_has_2d_errors,
    validate_publication,
)
from xdart.gui.tabs.static_scan.display_logic import Mode, compute_display_state
from xdart.gui.tabs.static_scan.display_publication import (
    PublicationDisplayAdapter,
    publication_availability,
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
        return publication_from_live_frame(DuckFrame(idx=int(label)))

    store.set_hydrator(hydrator)
    fresh = store.get_or_hydrate(5)
    assert calls == [5]
    assert fresh.view.has_1d                      # rehydrated + upserted
    # a hydrated publication short-circuits (bounds permitting)
    store2 = PublicationStore()
    store2.set_hydrator(hydrator)
    store2.upsert(publication_from_live_frame(DuckFrame(idx=6)))
    calls.clear()
    assert store2.get_or_hydrate(6).view.has_1d
    assert calls == []                            # no needless reload


def test_get_or_hydrate_rehydrates_tier1_thumbnail():
    # TIER-1 eviction (A2 fix): the payload (1D/2D/raw) is dropped but the
    # thumbnail is KEPT (semilight, raw_status="thumbnail").  get_or_hydrate MUST
    # rehydrate it.  Regression: the old guard counted the thumbnail as "heavy"
    # (_publication_has_heavy_payload) and short-circuited, so a tier-1 frame was
    # stuck on its thumbnail forever — the bug data_1d masked until Step 8b.
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


def test_publication_display_adapter_exposes_availability_and_int_plot_fallback():
    frame = DuckFrame(idx=9)
    frame.int_1d = IntegrationResult1D(
        radial=np.linspace(10.0, 20.0, 4),
        intensity=np.array([10.0, 20.0, 30.0, 40.0]),
        sigma=np.ones(4),
        unit="2th_deg",
    )
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))

    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    assert loaded_1d == {9}
    assert loaded_2d == {9}
    assert raw_avail[9] == {"has_raw": True, "has_thumbnail": True}

    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": False})()
        _plot_axis_info = [{"source": "1d", "slice_axis": None, "axis": None}]
        ui = type("UI", (), {
            "plotUnit": type("PlotUnit", (), {
                "currentIndex": staticmethod(lambda: 0),
                "currentText": staticmethod(lambda: "2θ (°)"),
            })(),
            "slice": type("Slice", (), {
                "isChecked": staticmethod(lambda: False),
            })(),
        })()

        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float) / metadata["monitor"]

    state = compute_display_state(
        mode=Mode.INT_1D,
        selected_ids=(9,),
        all_frame_index=[9],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )
    payload = PublicationDisplayAdapter(store, widget=_Widget()).plot_payload(state)

    # Step 5 FLIP: INT 1D now flows through the payload (was None pre-flip ->
    # legacy update_plot).  Data is already 2theta (2th_deg) and the request is
    # 2θ, so the native axis is used verbatim (no conversion); monitor-normalized.
    assert payload is not None
    np.testing.assert_allclose(payload.traces[0].x, np.linspace(10.0, 20.0, 4))
    np.testing.assert_allclose(
        payload.traces[0].y, np.array([10.0, 20.0, 30.0, 40.0]) / 100.0)
    assert (payload.axis_x.label, payload.axis_x.unit) == ("2θ", "°")


def test_publication_display_selected_labels_avoid_full_store_snapshot():
    class NoSnapshotStore(PublicationStore):
        def snapshot(self):  # pragma: no cover - exercised by failure
            raise AssertionError("selected-label display path copied full store")

    store = NoSnapshotStore()
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    loaded_1d, loaded_2d, raw_avail = publication_availability(
        store, labels=(2,))

    assert loaded_1d == {2}
    assert loaded_2d == {2}
    assert set(raw_avail) == {2}

    state = compute_display_state(
        mode=Mode.INT_1D,
        selected_ids=(2,),
        all_frame_index=[1, 2, 3],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )

    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": False})()
        _plot_axis_info = [{"source": "1d", "slice_axis": None, "axis": None}]
        ui = type("UI", (), {
            "plotUnit": type("PlotUnit", (), {
                "currentIndex": staticmethod(lambda: 0),
                "currentText": staticmethod(lambda: "Q (Å⁻¹)"),
            })(),
            "slice": type("Slice", (), {
                "isChecked": staticmethod(lambda: False),
            })(),
        })()

        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float)

    payload = PublicationDisplayAdapter(
        store, widget=_Widget(), labels=state.selected_ids,
    ).plot_payload(state)

    assert payload is not None
    assert len(payload.traces) == 1
    assert payload.traces[0].label == "scan_2"


def test_publication_display_adapter_builds_raw_and_cake_image_payloads():
    frame = DuckFrame(idx=11)
    frame.scan_info = {"monitor": 10.0}
    frame.map_raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    frame.bg_raw = 1.0
    frame.mask = np.array([0, 15])
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)

    class _Widget:
        global_mask = np.zeros((4, 4), dtype=bool)
        global_mask[1, 1] = True
        scan = type("Scan", (), {"name": "scan", "gi": False, "global_mask": global_mask})()
        bkg_map_raw = 0.0
        bkg_2d = 0.5

        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float) / metadata["monitor"]

    state = compute_display_state(
        mode=Mode.INT_2D,
        selected_ids=(11,),
        all_frame_index=[11],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )
    adapter = PublicationDisplayAdapter(store, widget=_Widget())

    raw = adapter.raw_image(state)
    cake = adapter.cake_image(state)

    expected_raw = np.asarray(frame.map_raw, dtype=float)
    expected_raw.ravel()[[0, 5, 15]] = np.nan
    expected_raw = ((expected_raw - 1.0) / 10.0)[::-1, :]
    assert raw is not None
    np.testing.assert_allclose(raw.image, expected_raw, equal_nan=True)
    assert raw.axis_x.label == "x"
    assert raw.axis_y.label == "y"

    expected_cake = frame.int_2d.intensity.T / 10.0 - 0.5
    assert cake is not None
    np.testing.assert_allclose(cake.image, expected_cake)


def _cake_state(store, idx):
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    return compute_display_state(
        mode=Mode.INT_2D, selected_ids=(idx,), all_frame_index=[idx],
        loaded_1d_keys=loaded_1d, loaded_2d_keys=loaded_2d, gi=False,
        plot_unit="q_A^-1", method="Single", unit_changed=False,
        prev_overlaid_ids=(), raw_availability=raw_avail,
        titles={}, generation=store.generation)


def test_cake_image_applies_imageunit_q_to_2theta_conversion():
    # The 2D-unit (imageUnit) Q↔2θ toggle is owned by cake_image now (so the
    # cake unit is consistent on every render, not only via the old
    # update_binned redraw).  Selecting "2θ-χ" over a Q-integrated cake converts
    # the radial axis values and relabels.
    from xdart.gui.tabs.static_scan.display_constants import Th, Chi
    frame = DuckFrame(idx=12)                      # int_2d.radial in Q (q_A^-1)
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))
    lam_m = 1.0e-10                                # 1 Å

    class _Combo:
        def __init__(self, text):
            self._text = text
        def currentText(self):
            return self._text

    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": False, "global_mask": None})()
        bkg_2d = 0
        def __init__(self, label):
            self.ui = type("UI", (), {"imageUnit": _Combo(label)})()
        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float)
        def _get_wavelength(self, frame=None):
            return lam_m

    state = _cake_state(store, 12)

    # Default "Q-χ": no conversion — axis stays Q.
    cake_q = PublicationDisplayAdapter(
        store, widget=_Widget(f"Q-{Chi}")).cake_image(state)
    assert cake_q is not None
    np.testing.assert_allclose(cake_q.axis_x.values, frame.int_2d.radial)

    # "2θ-χ": radial converted q -> 2θ and relabelled to degrees.
    cake_tth = PublicationDisplayAdapter(
        store, widget=_Widget(f"2{Th}-{Chi}")).cake_image(state)
    assert cake_tth is not None
    assert cake_tth.axis_x.unit == "°"        # degrees
    q = np.asarray(frame.int_2d.radial, dtype=float)
    lam_A = lam_m * 1e10
    expected = 2 * np.degrees(np.arcsin(np.clip(q * lam_A / (4 * np.pi), -1, 1)))
    np.testing.assert_allclose(cake_tth.axis_x.values, expected)
    # The cake image data itself is unchanged by the axis toggle.
    np.testing.assert_allclose(cake_tth.image, frame.int_2d.intensity.T)


def test_cake_image_gi_ignores_imageunit_toggle():
    # GI cakes keep their reciprocal-space axes verbatim (imageUnit disabled).
    from xdart.gui.tabs.static_scan.display_constants import Th, Chi
    frame = DuckFrame(idx=13)
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.0, 2.0, 4), azimuthal=np.linspace(0.0, 2.0, 3),
        intensity=np.ones((4, 3)), unit="qip_A^-1", azimuthal_unit="qoop_A^-1")
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))

    class _Combo:
        def currentText(self):
            return f"2{Th}-{Chi}"

    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": True, "global_mask": None})()
        bkg_2d = 0
        ui = type("UI", (), {"imageUnit": _Combo()})()
        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float)
        def _get_wavelength(self, frame=None):
            return 1.0e-10

    cake = PublicationDisplayAdapter(store, widget=_Widget()).cake_image(
        _cake_state(store, 13))
    assert cake is not None
    np.testing.assert_allclose(cake.axis_x.values, frame.int_2d.radial)   # verbatim


def test_gi_cake_axis_unit_is_angstrom_not_raw_key():
    # D1: the GI Q_ip/Q_oop cake axes show the unit Å⁻¹, not the raw integration
    # key qip_A^-1 / qoop_A^-1.  The label uses an HTML <sub> subscript (rendered
    # by pyqtgraph setLabel).
    from xdart.gui.tabs.static_scan.display_constants import AA_inv
    frame = DuckFrame(idx=14)
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.0, 2.0, 4), azimuthal=np.linspace(0.0, 2.0, 3),
        intensity=np.ones((4, 3)), unit="qip_A^-1", azimuthal_unit="qoop_A^-1")
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))

    class _Widget:
        scan = type("Scan", (), {"name": "scan", "gi": True, "global_mask": None})()
        bkg_2d = 0
        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float)

    cake = PublicationDisplayAdapter(store, widget=_Widget()).cake_image(
        _cake_state(store, 14))
    assert cake is not None
    assert cake.axis_x.label == "Q<sub>ip</sub>" and cake.axis_x.unit == AA_inv
    assert cake.axis_y.label == "Q<sub>oop</sub>" and cake.axis_y.unit == AA_inv
    assert cake.axis_x.values.shape == (4,)
    assert cake.axis_y.values.shape == (3,)


def test_plot_payload_delegates_to_integration_after_step5_flip():
    # Step 5 FLIP: plot_payload now delegates INT_1D/INT_2D to
    # integration_plot_payload.  The cases that PRE-flip fell back to the legacy
    # update_plot (2D-slice source, GI verbatim, Q<->2theta request) now return
    # a payload through plot_payload itself (== integration_plot_payload).
    frame = DuckFrame(idx=10)
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    state = compute_display_state(
        mode=Mode.INT_1D,
        selected_ids=(10,),
        all_frame_index=[10],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )

    def widget(*, source="1d", sliced=False, gi=False, text="Q (Å⁻¹)"):
        return type("Widget", (), {
            "scan": type("Scan", (), {"name": "scan", "gi": gi})(),
            "_plot_axis_info": [{"source": source, "slice_axis": "χ", "axis": "radial"}],
            "ui": type("UI", (), {
                "plotUnit": type("PlotUnit", (), {
                    "currentIndex": staticmethod(lambda: 0),
                    "currentText": staticmethod(lambda: text),
                })(),
                "slice": type("Slice", (), {
                    "isChecked": staticmethod(lambda: sliced),
                })(),
            })(),
            "normalize": staticmethod(lambda data, metadata: data),
        })()

    # native single: plot_payload returns a payload identical to the builder
    native = PublicationDisplayAdapter(store, widget=widget())
    p_native = native.plot_payload(state)
    direct = native.integration_plot_payload(state)
    assert p_native is not None and direct is not None
    np.testing.assert_allclose(p_native.traces[0].x, direct.traces[0].x)
    np.testing.assert_allclose(p_native.traces[0].y, direct.traces[0].y)
    # 2D-slice / GI verbatim / 2theta-request: all now return a payload
    # (previously None -> legacy update_plot fallback).
    assert PublicationDisplayAdapter(
        store, widget=widget(source="2d"),
    ).plot_payload(state) is not None
    assert PublicationDisplayAdapter(
        store, widget=widget(source="1d_2d", sliced=True),
    ).plot_payload(state) is not None
    assert PublicationDisplayAdapter(
        store, widget=widget(gi=True),
    ).plot_payload(state) is not None
    assert PublicationDisplayAdapter(
        store, widget=widget(text="2θ (°)"),
    ).plot_payload(state) is not None


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


def _two_cake_state(store):
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    # all_frame_index has a 3rd id so selecting (1, 2) is NOT "overall" —
    # otherwise the pre-existing overall guard (count != len(render_ids))
    # would short-circuit when the mismatched frame is skipped, masking the
    # skip behaviour we want to assert.
    return compute_display_state(
        mode=Mode.INT_2D,
        selected_ids=(1, 2),
        all_frame_index=[1, 2, 3],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Average",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )


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


def test_cake_image_does_not_blend_same_shape_different_axis_publications():
    # P2 #2: two cakes with the same (nchi, nq) shape but DIFFERENT chi axes
    # must NOT be averaged together — that's live↔batch↔reload axis drift the
    # publication contract is meant to catch.  Only the first (the accumulator
    # reference) is used; the mismatched frame is skipped.
    store = PublicationStore()
    store.upsert(publication_from_live_frame(
        _cake_frame(1, intensity=1.0, azimuthal=np.linspace(-90.0, 90.0, 3))))
    store.upsert(publication_from_live_frame(
        _cake_frame(2, intensity=3.0, azimuthal=np.linspace(-80.0, 100.0, 3))))

    cake = PublicationDisplayAdapter(store, widget=_cake_widget()).cake_image(
        _two_cake_state(store))

    assert cake is not None
    # Frame 1 only (1.0/10 = 0.1), NOT the blend (1.0+3.0)/2/10 = 0.2.
    np.testing.assert_allclose(cake.image, np.full((3, 4), 0.1))


def test_cake_image_blends_matching_axis_publications():
    # Control: identical axes DO average (behavior preserved).
    store = PublicationStore()
    store.upsert(publication_from_live_frame(
        _cake_frame(1, intensity=1.0, azimuthal=np.linspace(-90.0, 90.0, 3))))
    store.upsert(publication_from_live_frame(
        _cake_frame(2, intensity=3.0, azimuthal=np.linspace(-90.0, 90.0, 3))))

    cake = PublicationDisplayAdapter(store, widget=_cake_widget()).cake_image(
        _two_cake_state(store))

    assert cake is not None
    # Averaged: ((1.0 + 3.0) / 2) / 10 = 0.2.
    np.testing.assert_allclose(cake.image, np.full((3, 4), 0.2))


def test_publication_cake_background_transposes_legacy_pyfai_background():
    frame = DuckFrame(idx=1)
    legacy = np.arange(12, dtype=float).reshape(4, 3)
    frame.scan_info = {"monitor": 1.0}
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 4),
        azimuthal=np.linspace(-90.0, 90.0, 3),
        intensity=legacy,
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )
    store = PublicationStore()
    store.upsert(publication_from_live_frame(frame))
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    state = compute_display_state(
        mode=Mode.INT_2D,
        selected_ids=(1,),
        all_frame_index=[1],
        loaded_1d_keys=loaded_1d,
        loaded_2d_keys=loaded_2d,
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability=raw_avail,
        titles={},
        generation=store.generation,
    )
    widget = _cake_widget(monitor_norm=False)
    widget.bkg_2d = legacy

    cake = PublicationDisplayAdapter(store, widget=widget).cake_image(state)

    assert cake is not None
    np.testing.assert_allclose(cake.image, np.zeros((3, 4)))


def test_image_viewer_controller_owns_raw_preview_not_the_adapter():
    """The Image Viewer is a raw detector-file browser: ``ImageViewerController``
    builds its raw panel directly from the selected frame's stored detector
    array, applying NO processing mask, background subtraction or monitor
    normalization.  The publication adapter (which re-applies those, for the
    integration views) must not be the Image Viewer's source — routing the
    viewer through it blanked the panel after an Int 1D (XYE) run left
    normalization / Set-Background state on the widget.  The render is covered
    end-to-end in ``test_gui_modes_end_to_end.py``."""
    from threading import RLock
    from xdart.gui.tabs.static_scan.display_controllers import (
        ImageViewerController,
    )
    from xdart.gui.tabs.static_scan.display_logic import ImagePayload

    raw = np.array([[1.0, 65535.0], [3.0, 4.0]])     # uint16 ceiling, no NaN
    state = compute_display_state(
        mode=Mode.IMAGE_VIEWER,
        selected_ids=(1,),
        all_frame_index=[],
        loaded_1d_keys=set(),
        loaded_2d_keys={1},
        gi=False,
        plot_unit="q_A^-1",
        method="Single",
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability={1: {"has_raw": True, "has_thumbnail": False}},
        titles={},
        generation=7,
    )

    class _Widget:
        _viewer_is_xdart = False                      # standalone detector file
        data_lock = RLock()
        data_2d = {1: {"map_raw": raw, "thumbnail": None}}
        # A monitor/background that MUST be ignored by the raw browser.
        bkg_map_raw = np.array([[10.0, 10.0], [10.0, 10.0]])

        def normalize(self, data, metadata):
            return np.asarray(data, dtype=float) / 250.0

    payload = ImageViewerController().build_payload(_Widget(), state)

    assert payload.generation == 7
    assert payload.cake_image is None and payload.plot is None
    assert isinstance(payload.raw_image, ImagePayload)
    img = payload.raw_image.image
    # Standalone uint16 ceiling kept (not NaN-masked), finite, and NOT divided
    # by the monitor / background-subtracted.
    assert np.isfinite(img).all()
    assert np.nanmax(img) == 65535.0
    np.testing.assert_allclose(np.sort(img.ravel()), [1, 3, 4, 65535])
    assert payload.raw_image.axis_x.unit == "Pixels"


# ===================================================================== #
# Step 3: FrameRecord-backed publications (ADR-0003)
# ===================================================================== #

from xrd_tools.core import (  # noqa: E402
    DEFAULT_MODE_KEY,
    FrameRecord,
)
from xdart.modules.frame_publication import (  # noqa: E402
    legacy_to_canonical_1d,
    legacy_to_canonical_2d,
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
    f.gi_1d = {"qtotal": f.int_1d, "qip": qip}
    qchi = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 5), azimuthal=np.linspace(-90.0, 90.0, 4),
        intensity=np.ones((5, 4)), unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    f.gi_2d = {"gi2d": f.int_2d, "polar": qchi}
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
def test_legacy_to_canonical_mode_map_is_dimension_scoped():
    assert legacy_to_canonical_1d("qtotal") == "q_total"
    assert legacy_to_canonical_1d("qip") == "q_ip"
    assert legacy_to_canonical_1d("qoop") == "q_oop"
    assert legacy_to_canonical_1d("exit") == "exit_angle"
    assert legacy_to_canonical_2d("gi2d") == "qip_qoop"
    assert legacy_to_canonical_2d("polar") == "q_chi"          # 2D polar -> q_chi
    assert legacy_to_canonical_2d("exit2d") == "exit_angles"   # the coercer-gap key
    # already-canonical keys pass through
    assert legacy_to_canonical_1d("q_total") == "q_total"


def test_exit_angle_2d_mode_does_not_raise():
    f = DuckFrame(idx=9, gi=True)
    exit2d = IntegrationResult2D(
        radial=np.linspace(0.0, 5.0, 4), azimuthal=np.linspace(0.0, 90.0, 3),
        intensity=np.ones((4, 3)),
        unit="exit_angle_horz_deg", azimuthal_unit="exit_angle_vert_deg",
    )
    f.int_2d = exit2d
    f.gi_2d = {"exit2d": exit2d}
    rec = publication_from_live_frame(f).record
    assert rec.modes_2d == ("exit_angles",)
    assert rec.active_mode_2d == "exit_angles"


def test_reload_publication_carries_record(tmp_path):
    """publication_from_nexus_frame reads every persisted mode into the record."""
    recs = []
    for fi in range(2):
        f = _gi_multimode_frame(idx=fi)
        recs.append(publication_from_live_frame(f).record)
    p = str(tmp_path / "mm.nxs")
    with h5py.File(p, "w") as fh:
        write_frame_records(fh.create_group("entry"), recs)
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
                slice_on=False, center=0.0, width=1.0, wavelength_m=1e-10, gi=False):
    ui = SimpleNamespace(
        plotUnit=SimpleNamespace(currentIndex=lambda: 0,
                                 currentText=lambda: plot_unit_text),
        slice=SimpleNamespace(isChecked=lambda: slice_on),
        slice_center=SimpleNamespace(value=lambda: center),
        slice_width=SimpleNamespace(value=lambda: width),
    )
    return SimpleNamespace(
        scan=SimpleNamespace(name="scan", gi=gi),
        _plot_axis_info=[{"source": source, "slice_axis": None, "axis": axis}],
        ui=ui,
        normalize=lambda data, md: np.asarray(data, dtype=float)
        / ((md or {}).get("monitor", 1.0) or 1.0),
        _get_wavelength=lambda ref: wavelength_m,
    )


def _int_state(store, *, mode=Mode.INT_1D, method="Single", ids, gi=False,
               plot_unit="q_A^-1"):
    loaded_1d, loaded_2d, raw_avail = publication_availability(store)
    return compute_display_state(
        mode=mode, selected_ids=tuple(ids), all_frame_index=list(ids),
        loaded_1d_keys=loaded_1d, loaded_2d_keys=loaded_2d, gi=gi,
        plot_unit=plot_unit, method=method, unit_changed=False,
        prev_overlaid_ids=(), raw_availability=raw_avail, titles={},
        generation=store.generation,
    )


def _adapter(store, widget):
    return PublicationDisplayAdapter(store, widget=widget)


def test_integration_payload_native_single():
    frame = DuckFrame(idx=1)
    frame.scan_info = {"monitor": 2.0}
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, ids=(1,))
    payload = _adapter(store, _int_widget(plot_unit_text="q (Å⁻¹)")
                       ).integration_plot_payload(state)
    assert payload is not None and len(payload.traces) == 1
    np.testing.assert_allclose(payload.traces[0].x, frame.int_1d.radial)
    np.testing.assert_allclose(payload.traces[0].y, frame.int_1d.intensity / 2.0)


def test_integration_payload_q_to_2theta_conversion():
    frame = DuckFrame(idx=2)
    frame.scan_info = {"monitor": 1.0}
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, ids=(2,))
    w = _int_widget(plot_unit_text="2θ (°)", wavelength_m=1e-10)  # 1 Å
    payload = _adapter(store, w).integration_plot_payload(state)
    lam_A = 1.0
    q = np.asarray(frame.int_1d.radial)
    expected = 2 * np.degrees(np.arcsin(np.clip(q * lam_A / (4 * np.pi), -1, 1)))
    np.testing.assert_allclose(payload.traces[0].x, expected, rtol=1e-5)
    assert "2" in payload.axis_x.label or "th" in payload.axis_x.unit.lower()


def test_integration_payload_no_wavelength_keeps_native_axis():
    frame = DuckFrame(idx=3)
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, ids=(3,))
    w = _int_widget(plot_unit_text="2θ (°)", wavelength_m=None)
    payload = _adapter(store, w).integration_plot_payload(state)
    # no wavelength -> no conversion AND native (Q) axis kept (honest)
    np.testing.assert_allclose(payload.traces[0].x, frame.int_1d.radial)
    assert "q" in payload.axis_x.unit.lower() or payload.axis_x.label == "Q"


def test_integration_payload_gi_axis_verbatim():
    frame = DuckFrame(idx=4, gi=True)
    frame.int_1d = IntegrationResult1D(
        radial=np.linspace(-5.0, 5.0, 6), intensity=np.arange(6.0),
        sigma=None, unit="qip_A^-1",
    )
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, ids=(4,), gi=True)
    w = _int_widget(plot_unit_text="2θ (°)", gi=True)
    payload = _adapter(store, w).integration_plot_payload(state)
    assert payload is not None                       # GI no longer rejected
    np.testing.assert_allclose(payload.traces[0].x, frame.int_1d.radial)  # verbatim


def test_integration_payload_2d_slice_radial_transpose_guard():
    frame = DuckFrame(idx=5)
    frame.scan_info = {"monitor": 1.0}
    # intensity[radial, azimuthal] = radial index (independent of azimuthal)
    inten = np.tile(np.arange(4.0).reshape(4, 1), (1, 3))
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 4), azimuthal=np.linspace(-90.0, 90.0, 3),
        intensity=inten, unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, mode=Mode.INT_2D, ids=(5,))
    w = _int_widget(source="2d", axis="radial")
    payload = _adapter(store, w).integration_plot_payload(state)
    assert payload is not None
    # reduce over azimuthal -> per-radial value == radial index (the transpose
    # is correct; a wrong reduce-axis would average to a constant)
    np.testing.assert_allclose(payload.traces[0].y, np.arange(4.0))
    np.testing.assert_allclose(payload.traces[0].x, frame.int_2d.radial)


def test_integration_payload_2d_slice_window_and_label():
    frame = DuckFrame(idx=6)
    frame.scan_info = {"monitor": 1.0}
    # intensity[radial, azimuthal] = azimuthal index; azimuthal = [0,1,2,3]
    inten = np.tile(np.arange(4.0).reshape(1, 4), (4, 1))
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 4), azimuthal=np.array([0.0, 1.0, 2.0, 3.0]),
        intensity=inten, unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, mode=Mode.INT_2D, ids=(6,))
    # window center=0.5 width=0.6 -> azimuthal {0,1} selected -> mean 0.5
    w = _int_widget(source="2d", axis="radial", slice_on=True, center=0.5, width=0.6)
    payload = _adapter(store, w).integration_plot_payload(state)
    np.testing.assert_allclose(payload.traces[0].y, np.full(4, 0.5))
    assert "[" in payload.traces[0].label and "±" in payload.traces[0].label


def test_integration_payload_azimuthal_axis():
    frame = DuckFrame(idx=7)
    frame.scan_info = {"monitor": 1.0}
    inten = np.tile(np.arange(3.0).reshape(1, 3), (4, 1))  # [radial,azim]=azim
    frame.int_2d = IntegrationResult2D(
        radial=np.linspace(0.5, 3.0, 4), azimuthal=np.linspace(-90.0, 90.0, 3),
        intensity=inten, unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, mode=Mode.INT_2D, ids=(7,))
    w = _int_widget(source="2d", axis="azimuthal")
    payload = _adapter(store, w).integration_plot_payload(state)
    np.testing.assert_allclose(payload.traces[0].x, frame.int_2d.azimuthal)
    np.testing.assert_allclose(payload.traces[0].y, np.arange(3.0))


def test_integration_payload_overlay_waterfall_return_none():
    frame = DuckFrame(idx=8)
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    for method in ("Overlay", "Waterfall"):
        state = _int_state(store, ids=(8,), method=method)
        assert _adapter(store, _int_widget()).integration_plot_payload(state) is None


def test_plot_payload_defers_overlay_waterfall_to_legacy_after_flip():
    # Step 5 FLIP: plot_payload returns a payload for Single (delegating to
    # integration_plot_payload) but still returns None for Overlay/Waterfall so
    # render_display falls back to the legacy update_plot accumulator.
    frame = DuckFrame(idx=9)
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    adapter = _adapter(store, _int_widget())
    assert adapter.plot_payload(_int_state(store, ids=(9,), method="Single")) is not None
    for method in ("Overlay", "Waterfall"):
        assert adapter.plot_payload(
            _int_state(store, ids=(9,), method=method)) is None


def test_plot_payload_sum_average_emit_n_traces_collapsed_at_render():
    # Sum/Average go through the payload (NOT None): integration_plot_payload
    # emits one Trace per frame (un-reduced), exactly like legacy
    # get_frames_int_1d(rv='all'); the Sum/Average collapse happens at render in
    # update_1d_view (nanmean/nansum over the stacked rows).  So the payload
    # carries N traces and is non-None for Sum/Average.
    store = PublicationStore()
    for i in (0, 1, 2):
        f = DuckFrame(idx=i)
        f.scan_info = {"monitor": 1.0}
        f.int_1d = IntegrationResult1D(
            radial=np.linspace(0.5, 3.0, 6),
            intensity=np.full(6, float(i + 1)), sigma=None, unit="q_A^-1")
        store.upsert(publication_from_live_frame(f))
    adapter = _adapter(store, _int_widget())
    for method in ("Sum", "Average"):
        payload = adapter.plot_payload(
            _int_state(store, ids=(0, 1, 2), method=method))
        assert payload is not None
        assert len(payload.traces) == 3                       # un-reduced
        # all traces share the radial grid (collapse-ready)
        for tr in payload.traces:
            np.testing.assert_allclose(tr.x, payload.traces[0].x)
        # the render-level collapse would yield nanmean/nansum of [1,2,3]
        stack = np.vstack([tr.y for tr in payload.traces])
        np.testing.assert_allclose(np.nanmean(stack, 0), np.full(6, 2.0))
        np.testing.assert_allclose(np.nansum(stack, 0), np.full(6, 6.0))


def test_plot_payload_falls_back_when_store_evicted_even_if_render_ids_lists_it():
    # P1 (codex/other-claude review): the bounded store evicts older frames' 1D
    # arrays (max_heavy_items), but render_ids OR-merges the store with the
    # UNBOUNDED legacy data_1d (display_controllers._data_snapshot), so render_ids
    # can still LIST a store-evicted frame — a render_ids gate would wrongly pass.
    # The store-only available_1d_keys() gate must fall back to legacy update_plot
    # (which hydrates the full selection from disk); otherwise a whole-scan
    # Sum/Average/Overall silently drops the evicted frames.
    store = PublicationStore(max_heavy_items=1)
    for i in (0, 1, 2):
        f = DuckFrame(idx=i)
        f.scan_info = {"monitor": 1.0}
        f.int_1d = IntegrationResult1D(
            radial=np.linspace(0.5, 3.0, 6),
            intensity=np.full(6, float(i + 1)), sigma=None, unit="q_A^-1")
        store.upsert(publication_from_live_frame(f))
    assert not store.get(0).view.has_1d          # 0,1 thinned out of the store
    assert store.get(2).view.has_1d              # 2 stays resident
    adapter = _adapter(store, _int_widget())
    # Build the state as PRODUCTION does: loaded keys include the evicted frames
    # (the unbounded data_1d backstop), so render_ids == selected_ids — the
    # render_ids gate would PASS, but the store can't serve 0,1.
    for method in ("Average", "Sum", "Single"):
        state = compute_display_state(
            mode=Mode.INT_1D, selected_ids=(0, 1, 2), all_frame_index=[0, 1, 2],
            loaded_1d_keys={0, 1, 2}, loaded_2d_keys={0, 1, 2}, gi=False,
            plot_unit="q_A^-1", method=method, unit_changed=False,
            prev_overlaid_ids=(), raw_availability={}, titles={},
            generation=store.generation)
        assert set(state.render_ids) == {0, 1, 2}   # OR-merge masks the eviction
        assert adapter.plot_payload(state) is None  # store-only gate -> fall back
    # an all-resident selection still builds the payload (the flip applies).
    assert adapter.plot_payload(_int_state(store, ids=(2,), method="Single")) is not None


def test_integration_payload_gi_q_total_converts_to_2theta():
    # q_total is a |q| magnitude -> Bragg-convertible to 2θ exactly like a
    # standard scan, even with scan.gi True (unit-based guard, not the gi flag:
    # qtot_A^-1 is not is_gi_2d_units; qip/qoop/exit are).
    frame = DuckFrame(idx=10, gi=True)
    frame.scan_info = {"monitor": 1.0}
    frame.int_1d = IntegrationResult1D(
        radial=np.linspace(0.5, 3.0, 6), intensity=np.arange(6.0),
        sigma=None, unit="qtot_A^-1",
    )
    store = PublicationStore(); store.upsert(publication_from_live_frame(frame))
    state = _int_state(store, ids=(10,), gi=True)
    w = _int_widget(plot_unit_text="2θ (°)", gi=True, wavelength_m=1e-10)
    payload = _adapter(store, w).integration_plot_payload(state)
    q = np.asarray(frame.int_1d.radial)
    expected = 2 * np.degrees(np.arcsin(np.clip(q * 1.0 / (4 * np.pi), -1, 1)))
    np.testing.assert_allclose(payload.traces[0].x, expected, rtol=1e-5)


# ===================================================================== #
# Fork B: PublicationStore accumulates per-mode records (ADR-0003/0005)
# ===================================================================== #

from xrd_tools.core import FrameView, assert_framerecord_equivalent  # noqa: E402
from xdart.modules.frame_publication import _publication_has_heavy_payload  # noqa: E402


def _mode_pub(label, *, mode_1d=None, mode_2d=None, scale=1.0, generation=0,
              source_identity=None):
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
        source_identity=source_identity if source_identity is not None else str(label))


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

def test_record_keys_under_passed_active_mode_when_gi_dicts_empty():
    # The v2 reducer leaves gi_1d/gi_2d empty; the active_mode_* hint must key
    # the single-mode record under the REAL mode, not DEFAULT.
    rec = publication_from_live_frame(
        DuckFrame(idx=1, gi=True),
        active_mode_1d="q_oop", active_mode_2d="q_chi").record
    assert rec.modes_1d == ("q_oop",)
    assert rec.modes_2d == ("q_chi",)
    assert rec.active_mode_1d == "q_oop" and rec.active_mode_2d == "q_chi"


def test_record_stays_default_when_no_active_mode_passed():
    rec = publication_from_live_frame(DuckFrame(idx=1, gi=True)).record
    assert rec.active_mode_1d == DEFAULT_MODE_KEY
    assert rec.active_mode_2d == DEFAULT_MODE_KEY


def test_view_unchanged_by_active_mode_keying():
    f = DuckFrame(idx=2, gi=True)
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
    assert _same_source_id("", "frame_0001.tif")                                 # empty -> wildcard
