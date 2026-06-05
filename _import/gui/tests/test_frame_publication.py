from __future__ import annotations

import numpy as np
import pytest
import h5py

from ssrl_xrd_tools.core import (
    IntegrationResult1D,
    IntegrationResult2D,
    TwoDKind,
    assert_frameview_equivalent,
)
from ssrl_xrd_tools.io.nexus import write_integrated_stack

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
    store = PublicationStore(max_heavy_items=2)
    for idx in (1, 2, 3):
        store.upsert(publication_from_live_frame(DuckFrame(idx=idx)))

    assert store.labels() == (1, 2, 3)
    evicted = store.get(1)
    assert evicted is not None
    assert evicted.raw_ref is None
    assert evicted.raw_status == "evicted"
    assert not evicted.view.has_1d
    assert not evicted.view.has_2d
    assert evicted.view.thumbnail is None
    assert evicted.metadata_numeric["monitor"] == 100.0
    assert evicted.diagnostics.ok

    assert store.get(2).view.has_2d
    assert store.get(3).view.has_2d


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


def test_publication_display_adapter_exposes_availability_and_plot_payload():
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

    assert payload is not None
    assert payload.axis_x.label == "2θ"
    assert payload.axis_x.unit == "°"
    assert payload.traces[0].label == "scan_9"
    np.testing.assert_allclose(payload.traces[0].x, frame.int_1d.radial)
    np.testing.assert_allclose(payload.traces[0].y, frame.int_1d.intensity / 100.0)


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
    assert cake.axis_x.values.shape == (4,)
    assert cake.axis_y.values.shape == (3,)


def test_publication_display_adapter_falls_back_for_non_native_plot_modes():
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

    assert PublicationDisplayAdapter(
        store, widget=widget(source="2d"),
    ).plot_payload(state) is None
    assert PublicationDisplayAdapter(
        store, widget=widget(source="1d_2d", sliced=True),
    ).plot_payload(state) is None
    assert PublicationDisplayAdapter(
        store, widget=widget(gi=True),
    ).plot_payload(state) is None
    assert PublicationDisplayAdapter(
        store, widget=widget(text="2θ (°)"),
    ).plot_payload(state) is None


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
