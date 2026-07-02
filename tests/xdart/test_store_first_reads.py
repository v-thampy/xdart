# -*- coding: utf-8 -*-
"""Phase 5 A-Step-B: display reads consult FrameRecordStore first."""

from __future__ import annotations

from threading import RLock
from types import MethodType, SimpleNamespace

import numpy as np

from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_frame_view,
)
from xrd_tools.core import (
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    IntegrationResult2D,
    assert_frameview_equivalent,
)
from xrd_tools.session import FrameRecordStore


def _r1d(scale=1.0, *, unit="q_A^-1"):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial,
        intensity=intensity,
        sigma=np.sqrt(intensity),
        unit=unit,
    )


def _r2d(scale=1.0, *, unit="qip_A^-1", azimuthal_unit="qoop_A^-1"):
    radial = np.linspace(-1.0, 1.0, 3)
    azimuthal = np.linspace(-0.5, 0.5, 2)
    intensity = scale * np.arange(6, dtype=float).reshape(3, 2)
    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity,
        sigma=np.sqrt(intensity + 1.0),
        unit=unit,
        azimuthal_unit=azimuthal_unit,
    )


def _bind_store_first_methods(host):
    for name in (
        "_active_frame_record_modes",
        "_active_frame_record_store",
        "_publication_frame_view",
        "store_first_frame_view",
    ):
        setattr(host, name, MethodType(getattr(staticWidget, name), host))
    host._coerce_frame_label = staticWidget._coerce_frame_label
    for name in (
        "_selected_publication_views",
        "_first_present",
        "_publication_legacy_parts",
        "_display_publication_from_view",
        "_store_first_publication_for_display",
        "_snapshot_data",
    ):
        setattr(host, name, MethodType(getattr(DisplayDataMixin, name), host))
    host._display_hydration_should_block = (
        lambda allow_blocking_read=None: bool(allow_blocking_read)
    )
    return host


def _host(*, gi=False, store=None, publication_store=None, viewer_rows_1d=None, viewer_rows_2d=None):
    scan = SimpleNamespace(
        gi=gi,
        bai_1d_args={"gi_mode_1d": "q_total"} if gi else {},
        bai_2d_args={"gi_mode_2d": "qip_qoop"} if gi else {},
    )
    return _bind_store_first_methods(SimpleNamespace(
        scan=scan,
        data_lock=RLock(),
        viewer_rows_1d=viewer_rows_1d or {},
        viewer_rows_2d=viewer_rows_2d or {},
        publication_store=publication_store or PublicationStore(),
        _frame_record_store=store,
        wrangler=None,
        viewer_mode=None,
        display_generation=0,
    ))


def test_store_first_frame_view_matches_publication_and_ignores_viewer_rows_for_1d():
    metadata = {"i0": 2.0}
    view = FrameView.from_results(label=3, result_1d=_r1d(), metadata_raw=metadata)
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(view))
    publications = PublicationStore(max_heavy_items=None)
    publications.upsert(publication_from_frame_view(view))
    mirror = {3: SimpleNamespace(int_1d=_r1d(), scan_info=metadata)}
    host = _host(store=store, publication_store=publications, viewer_rows_1d=mirror)

    assert_frameview_equivalent(host.store_first_frame_view(3), view)

    host._frame_record_store = FrameRecordStore(max_heavy_items=None)
    assert_frameview_equivalent(host.store_first_frame_view(3), view)

    host.publication_store.clear()
    assert host.store_first_frame_view(3) is None


def test_store_first_frame_view_matches_publication_and_ignores_viewer_rows_for_gi_2d():
    metadata = {"i0": 2.0, "incident_angle": 0.18}
    view = FrameView.from_results(
        label=5,
        result_1d=_r1d(unit="qtot_A^-1"),
        result_2d=_r2d(),
        metadata_raw=metadata,
        incident_angle=0.18,
    )
    record = FrameRecord.from_view(
        view, mode_1d="q_total", mode_2d="qip_qoop"
    )
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(record)
    publications = PublicationStore(max_heavy_items=None)
    publications.upsert(publication_from_frame_view(view, record=record))
    frame_1d = SimpleNamespace(
        int_1d=_r1d(unit="qtot_A^-1"),
        gi_1d={"q_total": _r1d(unit="qtot_A^-1")},
        scan_info=metadata,
    )
    frame_2d = {
        "int_2d": _r2d(),
        "gi_2d": {"qip_qoop": _r2d()},
    }
    host = _host(
        gi=True,
        store=store,
        publication_store=publications,
        viewer_rows_1d={5: frame_1d},
        viewer_rows_2d={5: frame_2d},
    )

    assert_frameview_equivalent(
        host.store_first_frame_view(5, mode_1d="q_total", mode_2d="qip_qoop"),
        view,
    )
    assert_frameview_equivalent(
        host.store_first_frame_view(5, mode_1d="qtotal", mode_2d="gi2d"),
        view,
    )

    host._frame_record_store = FrameRecordStore(max_heavy_items=None)
    assert_frameview_equivalent(
        host.store_first_frame_view(5, mode_1d="q_total", mode_2d="qip_qoop"),
        view,
    )

    host.publication_store.clear()
    assert host.store_first_frame_view(
        5, mode_1d="q_total", mode_2d="qip_qoop") is None


def test_snapshot_data_uses_store_before_publication_and_ignores_viewer_rows():
    metadata = {"i0": 1.0}
    store_view = FrameView.from_results(
        label=7, result_1d=_r1d(scale=9.0), metadata_raw=metadata)
    publication_view = FrameView.from_results(
        label=7, result_1d=_r1d(scale=4.0), metadata_raw=metadata)
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(store_view))
    publications = PublicationStore(max_heavy_items=None)
    publications.upsert(publication_from_frame_view(publication_view))
    mirror = {7: SimpleNamespace(int_1d=_r1d(scale=1.0), scan_info=metadata)}
    host = _host(store=store, publication_store=publications, viewer_rows_1d=mirror)

    snapshot = host._snapshot_data([7])

    frame_1d, _frame_2d = snapshot[7]
    np.testing.assert_allclose(
        frame_1d.int_1d.intensity,
        store_view.intensity_1d,
    )

    host._frame_record_store = FrameRecordStore(max_heavy_items=None)
    snapshot = host._snapshot_data([7])
    frame_1d, _frame_2d = snapshot[7]
    np.testing.assert_allclose(
        frame_1d.int_1d.intensity,
        publication_view.intensity_1d,
    )

    host.publication_store.clear()
    assert host._snapshot_data([7]) == {}
