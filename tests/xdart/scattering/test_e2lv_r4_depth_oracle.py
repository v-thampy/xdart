"""Frozen depth oracle for the finite E2-LV-R4 responsibility correction."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
from threading import Event

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from xdart.gui.tabs.scattering import display_runtime
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_MIN
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.widgets.controls_panel import RangeRow
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.core.staging import browse_publication_max_items
from xrd_tools.io import read_frame_record
from xrd_tools.io.image_source import load_processed_raw_or_thumbnail
from xrd_tools.reduction import (
    FrameReduction, NexusSink, ReductionPlan, ReductionResult,
)
from xrd_tools.sources.image import TiffSeriesSource


def _owner(state: RunDisplayState, index: int):
    return state.add_artifact(
        Path(f"/run/artifact-{index}.nxs"),
        f"scan-{index}",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )


def _retain(
    state: RunDisplayState,
    owner,
    label: int,
    *,
    raw: np.ndarray | None = None,
    frame_mask_qualified: bool = False,
):
    image = (
        np.full((4, 4), label, dtype=np.uint16)
        if raw is None
        else np.asarray(raw)
    )
    axis = Axis(
        label="q",
        unit="1/angstrom",
        values=np.linspace(0.1, 1.0, 8),
    )
    view = FrameView(
        label=label,
        axis_1d=axis,
        intensity_1d=np.arange(8, dtype=float) + label,
        # Current heavy publications retain 2-D science; the sole light-1D
        # lease owns curves independently of this aggregate cake/raw budget.
        axis_2d_x=Axis(label="q", unit="1/angstrom", values=np.arange(4.0)),
        axis_2d_y=Axis(label="chi", unit="deg", values=np.arange(4.0)),
        intensity_2d=image.astype(float),
        raw=image,
        thumbnail=image.astype(float),
        source_path=f"/source/{owner.source_scan}.tif",
        source_frame_index=label,
    )
    record = FrameRecord.from_view(view)
    delta = state.append_navigation(
        owner.source_scan, str(owner.artifact), label
    )
    publication = FramePublication(
        replace(view, raw=None),
        record=record,
        source_identity=f"{view.source_path}#{label}",
        scan_key=owner.source_scan,
    )
    # The writer owns records and persistence qualification before the display
    # receives a publication; eviction must consult that real custody evidence.
    owner.records.upsert(
        record, source_identity=publication.source_identity, persisted=True,
    )
    state.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity=publication.source_identity,
        frame_mask_qualified=frame_mask_qualified,
    )
    return delta


def test_nine_artifacts_do_not_consume_eight_heavy_slots(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    state = RunDisplayState(RunIdentity(1, "nine-artifacts"), max_payload_items=2)
    state.configure(partition_count=9, npt=8, frame_bytes=16)

    owners = tuple(_owner(state, index) for index in range(9))

    assert len(owners) == 9
    assert state.residency_snapshot().limits.heavy == 8
    assert all(owner.records._max_heavy_items is None for owner in owners)


def test_uneven_artifacts_borrow_idle_capacity_under_each_run_limit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    monkeypatch.setattr(display_runtime, "THUMBNAIL_MAX_ITEMS", 20)
    monkeypatch.setattr(display_runtime, "live_record_store_max_items", lambda _npt: 20)
    monkeypatch.setattr(display_runtime, "browse_publication_max_items", lambda _npt: 20)
    state = RunDisplayState(RunIdentity(1, "borrow"), max_payload_items=2)
    state.configure(partition_count=3, npt=8, frame_bytes=16)
    first, second, _idle = (_owner(state, index) for index in range(3))

    for label in range(1, 9):
        _retain(state, first, label)
    for label in range(1, 3):
        _retain(state, second, label)

    snapshot = state.residency_snapshot()
    assert snapshot.heavy <= snapshot.limits.heavy
    assert snapshot.thumbnails <= snapshot.limits.thumbnails
    assert snapshot.browse <= snapshot.limits.browse
    assert snapshot.live <= snapshot.limits.live
    assert sum(
        first.records.has_heavy_payload(label)
        for label in first.records.labels()
    ) > snapshot.limits.heavy // 3


def test_navigation_capacity_covers_every_resident_browse_row() -> None:
    npt = 1_000
    capacity = browse_publication_max_items(npt)
    state = RunDisplayState(RunIdentity(1, "browse-capacity"), max_payload_items=2)
    state.configure(partition_count=1, npt=npt, frame_bytes=16)

    first = None
    for label in range(1, capacity + 1):
        delta = state.append_navigation("scan", "artifact", label)
        first = first or delta.appended

    assert state.navigation_capacity >= capacity
    assert len(state.catalog_snapshot().entries) == capacity
    assert first is not None and state.resolve_frame(first) is first


def test_catalog_operation_counter_rejects_tuple_copy_per_append() -> None:
    class CountedDeque(deque):
        yielded = 0

        def __iter__(self):
            for item in super().__iter__():
                type(self).yielded += 1
                yield item

    state = RunDisplayState(RunIdentity(1, "tuple-cost"), max_payload_items=2)
    state.catalog._entries = CountedDeque()
    for label in range(1, 501):
        state.append_navigation("scan", "artifact", label)

    assert CountedDeque.yielded <= 10


def test_frame_local_mask_matches_live_and_hydrated_durable_thumbnail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    frame_mask = np.zeros((2, 4), dtype=bool)
    frame_mask[0, 0] = True
    real_frame_for = TiffSeriesSource.frame_for

    def masked_frame_for(self, index):
        return replace(real_frame_for(self, index), mask=frame_mask)

    monkeypatch.setattr(TiffSeriesSource, "frame_for", masked_frame_for)
    qapp, page, lifecycle, executor, output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 19)), mask=None
    )
    shell, controller = lv_support._mounted(page)
    try:
        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        _disable_value_mask(qapp, shell)
        shell.run_controls.startButton.click()
        lv_support._completed_acquisition(qapp, page, lifecycle, executor)
        live = shell.scientific.raw.image.image
        assert live is not None and np.isnan(live[0, 1])

        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        lv_support._wait(
            qapp,
            lambda: (
                    shell.scientific.frame_selector.currentData() is first
                    and shell.scientific.title.text() == "tiny_0001.tif"
            ),
        )
        hydrated = shell.scientific.raw.image.image
        durable = read_frame_record(output, 1).active_view()
        assert hydrated is not None and np.isnan(hydrated[0, 1])
        assert durable.thumbnail is not None
        assert np.isnan(durable.thumbnail[0, 0])
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_historical_dense_integer_raw_keeps_native_mask_policy(
    tmp_path: Path,
    dtype,
) -> None:
    master = tmp_path / f"native_{np.dtype(dtype).name}_master.h5"
    raw = np.zeros((1, 100, 100), dtype=dtype)
    raw[0, 0, :2] = np.iinfo(dtype).max
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data", data=raw)

    processed = tmp_path / f"processed_{np.dtype(dtype).name}.nexus"
    sink = NexusSink(processed, overwrite=True, atomic=False)
    sink.begin(Scan("native-dtype", []), ReductionPlan(integration_2d=None))
    sink.write(
        ScanFrame(1, source_path=master, source_frame_index=0),
        FrameReduction(1, result_1d=IntegrationResult1D(
            radial=np.linspace(0.1, 1.0, 8), intensity=np.ones(8),
            sigma=None, unit="q_A^-1",
        )),
    )
    sink.finish(ReductionResult("native-dtype", {}, 1))

    loaded = load_processed_raw_or_thumbnail(
        processed, 1, preserve_raw_dtype=True
    )
    projected, mask_baked = display_runtime.project_detector_values(
        loaded.image, None, value_mask_enabled=True
    )

    assert loaded.source == "raw"
    assert loaded.image.dtype == raw.dtype
    assert mask_baked
    assert np.isnan(projected[0, :2]).all()


def _disable_value_mask(qapp, shell) -> None:
    threshold = next(
        row for row in shell.controls.findChildren(RangeRow)
        if tuple(row._low_path) == THRESHOLD_MIN
    )
    assert threshold._toggle[1].isChecked()
    threshold._toggle[1].click()
    lv_support._wait(qapp, lambda: shell.run_controls.startButton.isEnabled())


def test_static_mask_and_value_toggle_off_remain_exact_after_hydration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    static_mask = np.zeros((2, 4), dtype=bool)
    static_mask[0, 0] = True

    def write_saturated(path, *, offset=0):
        image = np.zeros((2, 4), dtype=np.uint16)
        image[0, 1] = np.iinfo(np.uint16).max
        fabio.tifimage.TifImage(data=image).write(str(path))

    from xrd_tools.io import frame_preview as preview_module

    loaded_dtypes: list[np.dtype] = []
    real_detector_read = preview_module.read_image

    def traced_loader(path, *args, **kwargs):
        loaded = real_detector_read(path, *args, **kwargs)
        loaded_dtypes.append(np.asarray(loaded).dtype)
        return loaded

    monkeypatch.setattr(lv_support, "_write_tiff", write_saturated)
    monkeypatch.setattr(preview_module, "read_image", traced_loader)
    qapp, page, lifecycle, executor, _output = lv_support._standard_page(
        monkeypatch,
        tmp_path,
        labels=tuple(range(1, 19)),
        mask=static_mask,
    )
    shell, controller = lv_support._mounted(page)
    try:
        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        _disable_value_mask(qapp, shell)
        shell.run_controls.startButton.click()
        lv_support._completed_acquisition(qapp, page, lifecycle, executor)
        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        lv_support._wait(
            qapp,
            lambda: (
                    shell.scientific.frame_selector.currentData() is first
                    and shell.scientific.title.text() == "tiny_0001.tif"
            ),
        )
        rendered = shell.scientific.raw.image.image
        assert rendered is not None
        assert np.isnan(rendered[0, 1])
        assert np.isfinite(rendered[1, 1])
        # Thumbnail-backed preview: zero detector-source reads by the accepted
        # one-open policy; the exact baked mask arrives via the persisted
        # thumbnail.  (Native-dtype fallback reads are owned by the frozen
        # no-thumbnail preview rows.)
        assert loaded_dtypes == []
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_public_close_retries_the_exact_context_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    qapp, page, lifecycle, executor, _output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 19)), mask=None
    )
    shell, controller = lv_support._mounted(page)
    from xdart.gui.tabs.scattering import hydration_transport

    entered = Event()
    release = Event()
    real_reader = hydration_transport.read_frame_preview

    def latched_reader(read_key, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return real_reader(read_key, **kwargs)

    executor._join_timeout = 0.01
    worker = None
    try:
        lv_support._wait(
            qapp, lambda: shell.run_controls.startButton.isEnabled()
        )
        shell.run_controls.startButton.click()
        lv_support._completed_acquisition(qapp, page, lifecycle, executor)
        monkeypatch.setattr(
            hydration_transport, "read_frame_preview", latched_reader
        )
        identity = controller.run_identity
        assert identity is not None
        run = executor._exact_run(identity)
        assert run is not None
        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        assert entered.wait(2.0)

        pending = page.close_workspace()
        assert run.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity
        worker = run.display.hydration_thread
        release.set()
        assert worker is not None
        worker.join(2.0)

        terminal = page.close_workspace()
        assert run.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert controller.run_identity is None
        assert controller.acquisition_context is None
    finally:
        release.set()
        if worker is not None:
            worker.join(2.0)
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
