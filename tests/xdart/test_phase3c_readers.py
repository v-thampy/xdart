# -*- coding: utf-8 -*-
"""Phase 3c: display readers resolve a frame's metadata store-first, with the
in-memory frames cache then the legacy data_1d mirror as fallbacks."""
from types import SimpleNamespace, MethodType

from xdart.gui.tabs.static_scan.display_plot import DisplayPlotMixin
from xdart.modules.frame_publication import (
    PublicationStore, publication_from_frame_view)
from xrd_tools.core import FrameView, numeric_metadata


def _holder(**kw):
    h = SimpleNamespace(**kw)
    h._frame_scan_info = MethodType(DisplayPlotMixin._frame_scan_info, h)
    return h


def _store_with(label, info):
    store = PublicationStore()
    store.upsert(publication_from_frame_view(
        FrameView.from_results(
            label=label, metadata_raw=info,
            metadata_numeric=numeric_metadata(info)),
        generation=store.generation))
    return store


def test_frame_scan_info_prefers_the_store():
    h = _holder(publication_store=_store_with(5, {"epoch": 100.0, "th": 0.2}),
                frames={5: SimpleNamespace(scan_info={"epoch": -1.0})},
                data_1d={5: SimpleNamespace(scan_info={"epoch": -2.0})})
    assert h._frame_scan_info(5)["epoch"] == 100.0      # store wins


def test_frame_scan_info_falls_back_to_frames_then_data_1d():
    # not in store -> in-memory frames cache
    h = _holder(publication_store=PublicationStore(),
                frames={7: SimpleNamespace(scan_info={"epoch": 9.0})},
                data_1d={})
    assert h._frame_scan_info(7)["epoch"] == 9.0
    # no store, not in frames -> legacy data_1d mirror
    h2 = _holder(publication_store=None, frames={},
                 data_1d={9: SimpleNamespace(scan_info={"epoch": 3.0})})
    assert h2._frame_scan_info(9)["epoch"] == 3.0


def test_frame_scan_info_empty_when_nothing_found():
    h = _holder(publication_store=None, frames={}, data_1d={})
    assert h._frame_scan_info(1) == {}
