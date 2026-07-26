"""X1 GUI-adoption Slice 2 — metadata + normalization consumers.

Migrates the frame-metadata popup and normalization-channel DISCOVERY onto the
accepted scan-qualified :class:`FrameProjection` (Slice-1 adapter), while render
math stays row-aligned (each trace normalized by its own ``MetadataRow``).

Production-wired: real ``staticWidget`` render/popup for the popup + channel
migration (with mutation guards that go red if production reads ``scan_data``);
real ``FrameRecordStore``/``project_frame`` for the row-aligned normalization and
the exact typed-conflict contract (S2-R3/R7).
"""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets
from PySide6.QtCore import QModelIndex

from tests.xdart._accepted_run import (  # noqa: E402
    accepted_run,
    container_source,
    gi_intent,
)
from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from xdart.modules.ewald.frame import LiveFrame
from xdart.modules.frame_publication import (
    publication_from_frame_view,
    publication_from_live_frame,
    publication_from_nexus_frame,
)
from xrd_tools.core import FrameRecord, FrameView, IntegrationResult1D
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.session import (
    CapabilityState,
    FrameRecordStore,
    project_frame,
)
from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _r1d(scale=1.0):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial, intensity=intensity, sigma=np.sqrt(intensity), unit="q_A^-1")


def _view(label, *, meta, source=("/data/loaded.nxs", None)):
    return FrameView.from_results(
        label=label, result_1d=_r1d(), metadata_raw=dict(meta),
        source_path=source[0],
        source_frame_index=label if source[1] is None else source[1])


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def _show_metawidget(qapp, metawidget):
    """Reparent the metadata widget's layout so its tableview is on screen
    (update() gates on tableview visibility — the production reparent pattern)."""
    frame = QtWidgets.QFrame()
    frame.setLayout(metawidget.layout)
    win = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(win).addWidget(frame)
    win.show()
    qapp.processEvents()
    assert metawidget.tableview.isVisible()
    return win


# --------------------------------------------------------------------------- #
# metadata popup: projection, not scan_data (S2-R6 mutation guard)
# --------------------------------------------------------------------------- #

def test_metadata_popup_shows_projection_not_scan_data(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    win = None
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        # The projection's row is the TRUTH for frame 0.
        display.publication_store.upsert(
            publication_from_frame_view(_view(0, meta={"i0": 111.0, "motor_x": 7.0})))
        # A deliberately stale/contradictory scan_data for the same frame.
        display.scan.scan_data = pd.DataFrame(
            {"i0": [999.0], "motor_x": [-1.0]}, index=[0])
        display.frame_ids = (0,)
        display.update()

        win = _show_metawidget(qapp, widget.metawidget)
        widget.metawidget.update()
        model = widget.metawidget.tableview.model()
        frame_df = model.dataFrame            # transposed: index=keys, columns=[label]
        # Mutation guard: production must show the PROJECTION values, never
        # scan_data's 999 / -1.  Restoring a scan_data fallback fails this.
        assert float(frame_df.loc["i0", 0]) == 111.0
        assert float(frame_df.loc["motor_x", 0]) == 7.0
    finally:
        if win is not None:
            win.close()
        widget.close()
        widget.deleteLater()


def test_scanned_motor_precedence_flows_through_projection(qapp, monkeypatch, tmp_path):
    # The composed per-frame row (scanned motor + counter) reaches the popup via
    # the projection with provider=None — precedence is baked in at ingestion.
    widget = _make_widget(monkeypatch, tmp_path)
    win = None
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.publication_store.upsert(
            publication_from_frame_view(_view(0, meta={"i0": 5.0, "th": 2.5})))
        display.frame_ids = (0,)
        display.update()
        win = _show_metawidget(qapp, widget.metawidget)
        widget.metawidget.update()
        frame_df = widget.metawidget.tableview.model().dataFrame
        assert float(frame_df.loc["th", 0]) == 2.5      # scanned motor present
        assert float(frame_df.loc["i0", 0]) == 5.0      # counter present
    finally:
        if win is not None:
            win.close()
        widget.close()
        widget.deleteLater()


def _project_publication(publication):
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(publication.record)
    return project_frame(store, publication.label)


def test_bluesky_metadata_projection_matches_live_batch_and_reload(tmp_path):
    """A real NXWriter motor/counter row survives all three production paths."""
    source = _write_bluesky_nxwriter(tmp_path / "bluesky_00001.nxs", n=3)
    reader = imageThread.__new__(imageThread)
    reader.meta_dir = None
    reader._eiger_metadata_cache = {}
    reader._bluesky_source_cache = {}
    reader.img_file = str(source)
    reader.run_configuration = accepted_run(
        source_spec=container_source(source))

    source_frame_index = 1
    label = source_frame_index + 1
    metadata = reader._frame_scan_info(reader.run_configuration, str(source), source_frame_index)
    assert {"hy", "i0"} <= metadata.keys()

    integrator = SimpleNamespace()
    live_frame = LiveFrame(
        label,
        np.ones((2, 2), dtype=np.float32),
        scan_info=dict(metadata),
        static=True,
        integrator=integrator,
    )
    live_frame.source_file = str(source)
    live_frame.source_frame_idx = source_frame_index
    live_frame.int_1d = _r1d()
    live = _project_publication(
        publication_from_live_frame(live_frame, include_2d=False))

    batch_worker = imageThread.__new__(imageThread)
    batch_worker.command = ""
    batch_worker.poni = None
    batch_worker.run_configuration = accepted_run(
        gi=gi_intent(incidence_motor="th", sample_orientation=4),
        source_spec=container_source(source))
    batch_worker._apply_threshold_inline = lambda _frozen, image: image
    batch_worker._resolve_frame_mask = lambda _frozen, scan, image: None
    batch_scan = SimpleNamespace(
        skip_2d=True,
        _cached_integrator=SimpleNamespace(),
    )
    [batch_frame] = imageThread._build_batch_frames(
        batch_worker, batch_worker.run_configuration,
        batch_scan,
        [(str(source), label, np.ones((2, 2), dtype=np.float32),
          dict(metadata), 0.0, 0.0)],
    )
    batch_frame.int_1d = _r1d()
    batch = _project_publication(
        publication_from_live_frame(batch_frame, include_2d=False))

    processed = tmp_path / "processed.nxs"
    with h5py.File(processed, "w") as h5:
        entry = h5.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[label],
            results_1d=[live_frame.int_1d],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([label]))
        for key, value in metadata.items():
            if isinstance(value, (int, float, np.number)):
                scan_data.create_dataset(key, data=np.array([float(value)]))
    reload = _project_publication(
        publication_from_nexus_frame(str(processed), label))

    expected = {key: float(metadata[key]) for key in ("hy", "i0")}
    for projection in (live, batch, reload):
        assert projection.metadata.raw["hy"] == pytest.approx(expected["hy"])
        assert projection.metadata.raw["i0"] == pytest.approx(expected["i0"])


# --------------------------------------------------------------------------- #
# channel discovery: projection, not scan_data; fail closed (S2-R5/R6)
# --------------------------------------------------------------------------- #

def test_channel_discovery_from_projection_not_scan_data(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.publication_store.upsert(
            publication_from_frame_view(_view(0, meta={"i0": 5.0, "temp": 300.0})))
        # scan_data carries a DIFFERENT channel that must NOT reach discovery.
        display.scan.scan_data = pd.DataFrame({"seconds": [1.0]}, index=[0])
        display.frame_ids = (0,)
        display.update()

        keys = display._norm_channel_discovery_keys()
        # Mutation guard: discovery keys come from the projection's numeric
        # metadata (i0, temp), never scan_data ("seconds").
        assert set(keys) == {"i0", "temp"}
        assert "seconds" not in keys
    finally:
        widget.close()
        widget.deleteLater()


def test_channel_discovery_fails_closed_without_current_projection(
        qapp, monkeypatch, tmp_path):
    # S2-R5: with no current scan-qualified projection (frame not in the store),
    # the authoritative channel set is EMPTY — never a prior/other scan's set,
    # and never scan_data's.
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.scan.scan_data = pd.DataFrame({"i0": [1.0]}, index=[0])  # has a channel
        display.frame_ids = (0,)                 # but no publication/record for 0
        display.update()
        assert display._norm_channel_discovery_keys() == []   # fail closed
    finally:
        widget.close()
        widget.deleteLater()


def test_channel_combo_tracks_new_selection_after_projection_pin(
        qapp, monkeypatch, tmp_path):
    """A selection change refreshes channels from the newly pinned frame.

    ``idxs_1d`` still describes the preceding render until ``get_idxs()`` runs.
    Channel discovery must therefore happen after the selection generation and
    projection pin, while reusing that pin instead of performing a second lookup.
    """
    widget = _make_widget(monkeypatch, tmp_path)
    win = None
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.publication_store.upsert(
            publication_from_frame_view(_view(0, meta={"i0": 2.0})))
        display.publication_store.upsert(
            publication_from_frame_view(_view(1, meta={"seconds": 3.0})))

        def channel_data():
            combo = display.ui.normChannel
            return [combo.itemData(row) for row in range(1, combo.count())]

        display.frame_ids = (0,)
        display.update()
        assert channel_data() == ["i0"]
        before = display._frame_projection_adapter.lookup_count

        display.frame_ids = (1,)
        display.update()

        assert display.idxs_1d == [1]
        assert channel_data() == ["seconds"]
        assert display._frame_projection_adapter.lookup_count == before + 1

        # The metadata popup consumes the same pinned projection rather than
        # issuing another store lookup for this selection/generation.
        win = _show_metawidget(qapp, widget.metawidget)
        widget.metawidget.update()
        assert display._frame_projection_adapter.lookup_count == before + 1
    finally:
        if win is not None:
            win.close()
        widget.close()
        widget.deleteLater()


def test_channel_choice_survives_transient_projection_gap_fail_closed(
        qapp, monkeypatch, tmp_path):
    """A live publication gap disables, but does not erase, normalization."""
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.publication_store.upsert(
            publication_from_frame_view(_view(0, meta={"i0": 2.0})))

        display.frame_ids = (0,)
        display.update()
        display.ui.normChannel.setCurrentIndex(1)
        assert display.ui.normChannel.currentData() == "i0"
        assert display.get_normChannel() == "i0"

        # Frame 1 is selected before its record is published.  The old choice is
        # visible but cannot be used as current-frame authority.
        display.frame_ids = (1,)
        display.update()
        assert display.ui.normChannel.currentData() == "i0"
        assert not display.ui.normChannel.isEnabled()
        assert display._norm_channel_signature == ()
        assert display.get_normChannel() is None
        np.testing.assert_allclose(
            display.normalize(np.array([2.0, 4.0]), {"i0": 2.0}),
            np.array([2.0, 4.0]),
        )

        # Once that same scan-qualified frame arrives, restore the preference
        # without relying on a selection-generation change.
        generation = display.display_generation
        display.publication_store.upsert(
            publication_from_frame_view(_view(1, meta={"i0": 4.0})))
        display.update()
        assert display.display_generation == generation
        assert display.ui.normChannel.isEnabled()
        assert display.ui.normChannel.currentData() == "i0"
        assert display.get_normChannel() == "i0"
        np.testing.assert_allclose(
            display.normalize(np.array([2.0, 4.0]), {"i0": 2.0}),
            np.array([1.0, 2.0]),
        )
    finally:
        widget.close()
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# render math: row-aligned normalization (S2-R3) + guard
# --------------------------------------------------------------------------- #

def _normalize_host(channel):
    host = SimpleNamespace()
    host.get_normChannel = lambda scan_data_keys=None: channel
    host.normalize = MethodType(DisplayDataMixin.normalize, host)
    return host


def test_normalize_is_row_aligned_per_frame():
    # S2-R3/R4: each trace is normalized by ITS OWN metadata row, never the
    # selected frame's.  Two Overlay rows with different monitors:
    host = _normalize_host("i0")
    out_a = host.normalize(np.array([8.0, 8.0]), {"i0": 2.0})
    out_b = host.normalize(np.array([8.0, 8.0]), {"i0": 4.0})
    assert out_a[0] == 4.0        # 8 / 2
    assert out_b[0] == 2.0        # 8 / 4 — its own row, not row A's


def test_normalize_missing_or_invalid_channel_is_a_no_op():
    host = _normalize_host("i0")
    # channel absent from THIS frame -> no normalization
    np.testing.assert_array_equal(host.normalize(np.array([8.0]), {"mon": 1.0}),
                                  np.array([8.0]))
    # non-finite / zero / negative value -> finite-positive guard, no norm
    np.testing.assert_array_equal(host.normalize(np.array([8.0]), {"i0": 0.0}),
                                  np.array([8.0]))
    np.testing.assert_array_equal(host.normalize(np.array([8.0]), {"i0": -3.0}),
                                  np.array([8.0]))


# --------------------------------------------------------------------------- #
# exact typed-conflict contract (S2-R7) + popup fails closed
# --------------------------------------------------------------------------- #

def test_conflict_projection_exact_error_state_and_empty_row():
    # S2-R7: assert the EXACT state + reason, and that the row is empty (never a
    # contradictory value).  Uses the provider path to force a stored/provider
    # metadata disagreement (the canonical conflict project_frame reports).
    store = FrameRecordStore()
    store.upsert(FrameRecord.from_view(
        _view(0, meta={"i0": 11.0}, source=("/data/raw.nxs", 1))))
    provider = SimpleNamespace(
        metadata_for=lambda idx: {"i0": 42.0},
        complete_metadata_for=lambda idx: {"i0": 42.0},
        frame_count=lambda: 2,
        motors=lambda: {},
        scan_table=lambda: {},
        constants=lambda: {},
        wavelength=lambda: None,
    )
    projected = project_frame(store, 0, provider=provider)
    assert projected.capabilities.metadata.state is CapabilityState.ERROR
    assert "stored/provider metadata disagree" in projected.capabilities.metadata.reason
    assert not projected.metadata            # empty row — no contradictory value


def test_metadata_popup_fails_closed_on_empty_projection(qapp, monkeypatch, tmp_path):
    # When the current projection has no usable row (absent/conflict), the popup
    # shows an EMPTY table — never a scan_data fallback value (S2-R6/R7).
    widget = _make_widget(monkeypatch, tmp_path)
    win = None
    try:
        display = widget.displayframe
        display.scan.name = "loaded"
        display.scan.scan_data = pd.DataFrame({"i0": [999.0]}, index=[0])
        display.frame_ids = (0,)                 # no publication/record -> absent
        display.update()
        win = _show_metawidget(qapp, widget.metawidget)
        widget.metawidget.update()
        model = widget.metawidget.tableview.model()
        assert model.rowCount(QModelIndex()) == 0   # empty, not scan_data's i0=999
    finally:
        if win is not None:
            win.close()
        widget.close()
        widget.deleteLater()
