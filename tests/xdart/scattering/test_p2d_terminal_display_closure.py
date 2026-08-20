from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import threading
import time

import h5py
import numpy as np
import pytest
import tifffile
from pyqtgraph import QtCore, QtWidgets

from xdart import _gui_main
import xdart.gui.tabs.scattering.context_controller as controller_module
import xdart.gui.tabs.scattering.hydration_transport as transport_module
import xdart.gui.tabs.scattering.scientific_view as scientific_view_module
import xdart.gui.tabs.scattering.shell_projection as projection_module
from xdart.gui.pages.catalog import SCATTERING_WORKSPACE_PAGE
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.context_runtime import _ContextRuntime
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardDisplayPayload
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, FrameNavigationProjection, HeavyProjection,
    ScientificProjection, SlicePin, TraceProjection,
)
from xdart.modules.display_context import CommitGate, HydrationOwner
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind, assert_frameview_equivalent
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io import read_frame_record, read_frame_view
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.io.nexus_record import ensure_frames_container, stamp_source_base, write_frame_record
from xrd_tools.session import (
    Light1DBufferLayout, Light1DLayout, Light1DModeData, Light1DModeLayout,
    Light1DRecord, SessionResourceAuthority, acquire_light_1d_retention,
)
from xrd_tools.session.hydration import HydrationCompletion, HydrationOutcome

from tests.xdart.scattering import test_p2c_common_viewer_mount as p2c


@pytest.fixture(autouse=True)
def _mutation(monkeypatch):
    mutation = os.environ.get("P2D_MUTATION", "")
    if mutation == "M01":
        original = _ContextRuntime._select

        def stale(self, context):
            prior = self.selection
            selected = original(self, context)
            if prior is not None and len(self.retained_contexts) > 1:
                self._selection = prior
                return prior
            return selected

        monkeypatch.setattr(_ContextRuntime, "_select", stale)
    elif mutation == "M02":
        original = projection_module.build_scientific_projection

        def leak(payloads, navigation, resident, preferences, notice, *args, **kwargs):
            result = original(payloads, navigation, resident, preferences, notice, *args, **kwargs)
            return replace(result, slice_pins=preferences.slice_pins)

        monkeypatch.setattr(projection_module, "build_scientific_projection", leak)
    elif mutation == "M03":
        owner_type = controller_module._TwoDViewerOwner
        original = owner_type.complete

        def accept_foreign(self, completion):
            current = self.request_token
            if completion.token is not current:
                self.request_token = completion.token
                original(self, completion)
                self.request_token = current
            else:
                original(self, completion)

        monkeypatch.setattr(owner_type, "complete", accept_foreign)
    elif mutation == "M04":
        original = Light1DLayout.shared_bytes.fget
        monkeypatch.setattr(Light1DLayout, "shared_bytes", property(lambda self: 513 * original(self)))
    elif mutation == "M05":
        original = RunDisplayState.request_full

        def evict_first(self, *args, **kwargs):
            self.invalidate_full_demand(clear_raw=True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(RunDisplayState, "request_full", evict_first)
    elif mutation == "M06":
        original = ScientificView._rebuild_frames

        def rescan(self, frames, selected):
            for index in range(self.frame_selector.count()):
                self.frame_selector.itemData(index)
            return original(self, frames, selected)

        monkeypatch.setattr(ScientificView, "_rebuild_frames", rescan)
    elif mutation == "M07":
        original = controller_module.release_browse

        def suppress(*args, **kwargs):
            receipt = original(*args, **kwargs)
            return replace(receipt, cleanup_status=CleanupStatus.CLEANED)

        monkeypatch.setattr(controller_module, "release_browse", suppress)
    elif mutation == "M08":
        monkeypatch.setattr(ContextController, "_retire_viewer_2d_standalone", lambda self: True)


@pytest.fixture
def mount(tmp_path, monkeypatch):
    value = p2c._Mount(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        value.finish()


def _wait(predicate, timeout=12.0):
    deadline = time.monotonic() + timeout
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _view(label, value, *, epoch=None, raw=True):
    image = np.arange(12.0).reshape(3, 4) + value
    metadata = {} if epoch is None else {"epoch": float(epoch)}
    return FrameView(
        label, axis_1d=Axis("Q", "q_A^-1", values=np.linspace(0.1, 0.4, 4)),
        intensity_1d=np.arange(4.0) + value,
        sigma_1d=np.full(4, 0.25),
        axis_2d_x=Axis("Q", "q_A^-1", values=np.linspace(0.1, 0.4, 4)),
        axis_2d_y=Axis("chi", "chi_deg", values=np.linspace(-1.0, 1.0, 3)),
        intensity_2d=image, two_d_kind=TwoDKind.Q_CHI,
        raw=image if raw else None, thumbnail=image[::2, ::2],
        source_path=f"/data/scan_{label:04d}.tif", source_frame_index=label,
        metadata_numeric=metadata,
    )


def _retain(display, owner, label, value):
    view = _view(label, value)
    record = FrameRecord.from_view(view)
    source = f"{view.source_path}#{label}"
    owner.records.upsert(record, source_identity=source)
    publication = FramePublication(view, record=record, source_identity=source, scan_key=owner.source_scan)
    delta = display.append_navigation(owner.source_scan, str(owner.artifact), label)
    display.retain_frame(owner, delta.appended, record, publication,
                         source_identity=source, frame_mask_qualified=False)
    owner.publications.upsert(publication)
    display.put_payload(StandardDisplayPayload(0, delta.appended,
                        f"Standard · {owner.source_scan} · frame {label}", view))
    return delta


def _extend_acquisition(mount):
    display = mount.acquisition.publication_store
    owner = next(iter(display.artifacts.values()))
    for label in (2, 3):
        assert mount.controller.accept_navigation(
            _retain(display, owner, label, float(label)),
            plot_mode="Overlay", follow_latest=True,
        )


def _browse_three(mount):
    request = mount.controller.begin_browse("/processed/browse.b.nxs")
    _, context = p2c._browse(request.token, request.load_generation, request=request)
    for label in (2, 3):
        view = _view(label, 20.0 + label)
        record = FrameRecord.from_view(view)
        source = f"/processed/b.nxs#{label}"
        context.record_store.upsert(record, source_identity=source, persisted=True)
        context.publication_store.upsert(FramePublication(
            view, record=record, source_identity=source, scan_key=context.scan_key))
        context.frame_ids.append(label)
    mount.loader.complete(context)
    assert mount.controller.poll_browse() is not None
    return context


def _science(controller, mode):
    preferences = ScientificPreferences(plot_mode=mode)
    payloads = controller.project_navigation(preferences=preferences, processing_mode="Int 1D")
    return projection_module.build_scientific_projection(
        payloads, controller.navigation, controller.resident_frame_keys,
        preferences, "", processing_mode="Int 1D",
        norm_aggregate=controller.norm_aggregate,
    )


def _mode_fingerprint(controller, frames):
    current = controller.navigation.current
    result = []
    for mode in ("Single", "Overlay", "Waterfall"):
        selected = (current,) if mode == "Single" else frames
        assert controller.select_navigation(current, selected)
        scientific = _science(controller, mode)
        assert scientific.plot_mode == mode
        assert len(scientific.traces) == (1 if mode == "Single" else len(frames))
        if mode != "Single":
            assert controller.commit_navigation_projection(
                tuple(trace.frame for trace in scientific.traces))
        result.append((mode, scientific.title, tuple(
            (trace.frame, trace.axis.label, trace.axis.unit,
             tuple(trace.axis.values), tuple(trace.intensity))
            for trace in scientific.traces)))
    return tuple(result)


def _write_result(root, labels):
    root.mkdir(parents=True, exist_ok=True)
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    raw_path = root / "raw.tif"
    tifffile.imwrite(raw_path, raw)
    processed = root / "result.nexus"
    radial, azimuthal = np.array([0.1, 0.2, 0.3]), np.array([-1.0, 1.0])
    one = [IntegrationResult1D(radial, np.array([1.0, 2.0, 3.0]) + label,
                              np.full(3, 0.5), "q_A^-1") for label in labels]
    two = [IntegrationResult2D(radial, azimuthal,
                              np.arange(6.0).reshape(3, 2) + label,
                              unit="q_A^-1", azimuthal_unit="chi_deg") for label in labels]
    with h5py.File(processed, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(entry, frame_indices=labels, results_1d=one, results_2d=two)
        base = stamp_source_base(entry, root)
        frames = ensure_frames_container(entry)
        for label in labels:
            write_frame_record(frames, f"frame_{label:04d}", thumbnail=raw[::2, ::2],
                               source_path=raw_path, source_frame_index=0, source_base=base)
    return processed, raw_path


def _cold_browse(root, labels=(1, 2, 3), max_items=32, *, acquisition=False):
    processed, _ = _write_result(root, list(labels))
    loader = BrowseLoader(max_items=max_items)
    controller = ContextController(
        lifecycle=ScatteringCoordinator(), executor=object(), browse_loader=loader,
        projection=ContextProjection())
    identity = context = None
    if acquisition:
        identity, context = p2c._acquisition()
        controller._runtime.adopt_acquisition(identity, context)
    request = controller.begin_browse(str(processed))
    outcome = None

    def loaded():
        nonlocal outcome
        outcome = controller.poll_browse()
        return outcome is not None

    assert _wait(loaded, 20.0)
    assert outcome.request is request and controller.browse_context is not None
    return controller, loader, context, controller.browse_context, processed, identity


def _select_label(controller, label, *, all_frames=False):
    frame = next(item for item in controller.frame_keys if item.local_frame_label == label)
    assert controller.select_navigation(frame, controller.frame_keys if all_frames else (frame,))
    return frame


def _light_layout():
    axis = Light1DBufferLayout(4, 8, "q-axis", "<f8", shared=True)
    return Light1DLayout((
        Light1DModeLayout("raw", axis,
                         Light1DBufferLayout(4, 8, "raw-y", "<f8"),
                         Light1DBufferLayout(4, 8, "raw-s", "<f8")),
        Light1DModeLayout("bg", axis,
                         Light1DBufferLayout(4, 4, "bg-y", "<f4")),
    ), "bg")


def _light_record(label, axis):
    return Light1DRecord(label, 3, "bg", {
        "raw": Light1DModeData(axis, np.full(4, label, dtype=np.float64),
                               np.full(4, label / 10, dtype=np.float64)),
        "bg": Light1DModeData(axis, np.full(4, label, dtype=np.float32)),
    }, {"source": "result.nexus"})


def test_modes_selection_navigation_and_restore_across_contexts(mount):
    _extend_acquisition(mount)
    controller = mount.controller
    a_frame = controller.frame_keys[1]
    assert controller.select_navigation(a_frame, (a_frame,))
    a_selection = controller.selection
    a_frames = controller.frame_keys[1:]
    a_modes = _mode_fingerprint(controller, a_frames)
    browse = _browse_three(mount)
    b_frame = controller.frame_keys[1]
    assert controller.select_navigation(b_frame, (b_frame,))
    b_selection = controller.selection
    b_frames = controller.frame_keys[1:]
    b_modes = _mode_fingerprint(controller, b_frames)
    assert a_frame.run_identity is b_frame.run_identity
    assert (a_selection.context_token, a_selection.owner) != (b_selection.context_token, b_selection.owner)
    assert a_frame.source_scan != b_frame.source_scan and a_frame is not b_frame

    restored_a = controller.select_acquisition()
    assert restored_a.display_generation > b_selection.display_generation
    assert controller.navigation.current is a_frame
    assert _mode_fingerprint(controller, a_frames) == a_modes
    restored_b = controller.select_browse()
    assert restored_b.display_generation > restored_a.display_generation
    assert restored_b.names(browse) and controller.navigation.current is b_frame
    assert _mode_fingerprint(controller, b_frames) == b_modes


def test_slice_pin_shared_axis_and_metadata_survive_replacement():
    old_identity, new_identity = RunIdentity(1, "old"), RunIdentity(2, "new")
    old = DisplayFrameKey(old_identity, "scan", "/same/result.nexus", 7, 1)
    new = DisplayFrameKey(new_identity, "scan", "/same/result.nexus", 7, 1)
    old_payload = StandardDisplayPayload(0, old, "old", _view(7, 1.0, epoch=1.0))
    new_payload = StandardDisplayPayload(0, new, "new", _view(7, 2.0, epoch=2.0))
    old_pin, new_pin = SlicePin(old, "Q", 0.0, 2.0), SlicePin(new, "Q", 0.0, 2.0)
    preferences = ScientificPreferences(
        plot_axis="Q", plot_mode="Overlay", share_axis=True,
        slice_enabled=True, slice_center=0.0, slice_width=2.0,
        slice_pins=(old_pin, new_pin),
    )
    navigation = FrameNavigationProjection((new,), new, (new,))
    projected = projection_module.build_scientific_projection(
        (old_payload, new_payload), navigation, frozenset({new}), preferences, "")
    assert projected.slice_pins == (new_pin,)
    assert len(projected.pinned_traces) == 1
    trace = projected.pinned_traces[0].trace
    assert trace.frame is new and trace.epoch == 2.0
    assert (trace.axis.label, trace.axis.unit) == ("Q", "q_A^-1")
    np.testing.assert_allclose(trace.axis.values, new_payload.view.axis_1d.values)
    assert projected.share_axis and projected.slice_enabled
    assert projected.heavy is not None and projected.heavy.frame is new


def test_deep_selection_save_reload_equivalence_and_foreign_late_inert(mount, tmp_path):
    processed, _ = _write_result(tmp_path / "deep", list(range(1, 514)))
    saved = read_frame_view(processed, 513)
    reloaded = read_frame_view(processed, 513)
    assert_frameview_equivalent(saved, reloaded)
    identity = RunIdentity(1, "deep")
    frame = DisplayFrameKey(identity, "deep", str(processed), 513, 513)
    payload = StandardDisplayPayload(0, frame, "deep:513", reloaded)
    navigation = FrameNavigationProjection((frame,), frame, (frame,))
    science = projection_module.build_scientific_projection(
        (payload,), navigation, frozenset({frame}), ScientificPreferences(), "")
    assert len(science.traces) == 1 and science.traces[0].frame is frame
    np.testing.assert_allclose(science.traces[0].intensity, reloaded.intensity_1d)

    mount.open2()
    owner = mount.controller._viewer_2d
    stale_token = owner.request_token
    mount.done2.clear()
    ScatteringWorkspace._choose_viewer_2d_file(mount.page, reload=True)
    mount.wait("2d")
    before = (owner.context, owner.catalog, owner.frame, owner.request_token,
              owner.loading, owner.diagnostic, mount.controller.selection,
              mount.controller.navigation, mount.controller._runtime._display_generation)
    owner.complete(HydrationCompletion(stale_token, HydrationOutcome.FAILED, "foreign late"))
    after = (owner.context, owner.catalog, owner.frame, owner.request_token,
             owner.loading, owner.diagnostic, mount.controller.selection,
             mount.controller.navigation, mount.controller._runtime._display_generation)
    assert after == before


def test_513_light_rows_account_unique_and_shared_bytes_and_evict_deterministically():
    layout = _light_layout()
    full_bytes = layout.shared_bytes + 513 * layout.per_row_unique_ndarray_bytes
    authority = SessionResourceAuthority(capacity_bytes=full_bytes)
    lease = acquire_light_1d_retention(
        authority, owner="full", generation=3, layout=layout,
        requested_rows=513, compatibility_byte_ceiling=full_bytes,
        gui_thread_id=threading.get_ident())
    axis = np.linspace(0.0, 1.0, 4)
    for label in range(513):
        lease.retain(_light_record(label, axis), grant_id=lease.grant_id,
                     generation=lease.generation)
    assert layout.shared_bytes == axis.nbytes
    assert lease.reserved_ndarray_bytes == full_bytes and lease.keys() == tuple(range(513))
    assert lease._roots[0]["q-axis"] is lease._roots[512]["q-axis"]
    assert lease._roots[0]["raw-y"] is not lease._roots[512]["raw-y"]

    reduced_bytes = layout.shared_bytes + 3 * layout.per_row_unique_ndarray_bytes
    reduced = acquire_light_1d_retention(
        SessionResourceAuthority(capacity_bytes=reduced_bytes), owner="reduced",
        generation=3, layout=layout, requested_rows=513,
        compatibility_byte_ceiling=reduced_bytes, gui_thread_id=threading.get_ident())
    for label in range(5):
        reduced.retain(_light_record(label, axis), grant_id=reduced.grant_id,
                       generation=reduced.generation)
    assert reduced.row_cap == 3 and reduced.keys() == (2, 3, 4)
    assert reduced.evicted == (0, 1)
    assert authority.snapshot().categories == {"light_1d": full_bytes}
    lease.release(reason="test"); reduced.release(reason="test")


def test_retained_hit_is_zero_io_and_evicted_miss_is_async(monkeypatch, tmp_path):
    processed, _raw_path = _write_result(tmp_path / "async", [1, 2])
    state = RunDisplayState(RunIdentity(1, "async"), max_payload_items=2)
    state.configure(partition_count=1, npt=4, frame_bytes=32)
    owner = state.add_artifact(processed, "async", mask=None,
                               mask_saturation=False, measurement_mode="Standard")
    state.bind_transport(event_sink=lambda _event: None)
    keys = {}
    for label in (1, 2):
        record = read_frame_record(processed, label)
        view = record.active_view()
        source = f"{view.source_path}#{view.source_frame_index}"
        owner.records.upsert(record, source_identity=source)
        modes = tuple([("1d", mode) for mode in record.results_1d] +
                      [("2d", mode) for mode in record.results_2d])
        owner.records.replace_projection(label, hydratable=modes, durable=modes)
        delta = state.append_navigation("async", str(processed), label)
        keys[label] = delta.appended
    view = read_frame_view(processed, 2)
    record = FrameRecord.from_view(view)
    publication = FramePublication(view, record=record, source_identity=f"{_raw_path}#0", scan_key="async")
    state.retain_frame(owner, keys[2], record, publication,
                       source_identity=publication.source_identity, frame_mask_qualified=False)
    gate = CommitGate()
    hydration_owner = HydrationOwner("ctx", "async", str(processed), gate.epoch)
    assert state.request_full(keys[2], 0, owner=hydration_owner, commit_gate=gate) is not None
    assert _wait(lambda: state.transport.queued_token is None and
                 state.transport.active_token is None)
    assert owner.publications.get(2).view.raw is not None
    entered, release, reads = threading.Event(), threading.Event(), []
    original = transport_module.read_frame_preview

    def held(read_key, **kwargs):
        reads.append(threading.current_thread().name)
        entered.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(transport_module, "read_frame_preview", held)
    try:
        assert state.request_full(keys[2], 1, owner=hydration_owner, commit_gate=gate) is not None
        assert reads == [] and state.full_raw_status(keys[2])[:2] == (True, False)
        state.invalidate_full_demand(clear_raw=True)
        assert state.request_full(keys[2], 2, owner=hydration_owner, commit_gate=gate) is not None
        assert entered.wait(10.0) and all(name != threading.main_thread().name for name in reads)
        assert state.full_raw_status(keys[2])[1] is True
        release.set()
        assert _wait(lambda: state.transport.queued_token is None and
                     state.transport.active_token is None)
        assert reads == ["scattering-preview-transport"]
        assert owner.publications.get(2).view.raw is not None
    finally:
        release.set(); gate.cancel(); state.retire(join_timeout=1.0)


def test_3621_history_show_all_overlay_waterfall_work_and_resources_stay_bounded(monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    identity = RunIdentity(1, "history")
    frames = tuple(DisplayFrameKey(identity, "scan", "result.nexus", index + 1, index + 1)
                   for index in range(3621))
    values = np.linspace(0.1, 4.0, 16)
    axis = AxisProjection(values, "Q", "Å⁻¹")
    traces = tuple(TraceProjection(frame, axis, np.sin(values) + index, f"scan:{index + 1}")
                   for index, frame in enumerate(frames))
    available = frozenset(frames[-2:])
    first_heavy = HeavyProjection(frames[-2], raw=np.ones((2, 2)))
    last_heavy = HeavyProjection(frames[-1], raw=np.ones((2, 2)) * 2)
    first = ScientificProjection(traces=traces[:-1], heavy=first_heavy,
                                 heavy_available=available, plot_mode="Overlay", live_update=True)
    view = ScientificView()
    try:
        view.reconcile(first, FrameNavigationProjection(frames[:-1], frames[-2], frames[:-1]),
                       completed=3620, total=3621, detail="Processing")
        operations = view._selector_operations
        probes = []
        original_item = view.frame_selector.itemData

        def counted(index, *args):
            probes.append(index)
            return original_item(index, *args)

        monkeypatch.setattr(view.frame_selector, "itemData", counted)
        delta = replace(first, traces=(traces[-1],), heavy=last_heavy)
        all_navigation = FrameNavigationProjection(frames, frames[-1], frames)
        view.reconcile(delta, all_navigation, completed=3621, total=3621, detail="Ready")
        assert len(probes) <= 4
        assert view._selector_operations - operations == 1
        assert view.frame_selector.count() == 3621 and view.trace_history_keys == frames
        assert view.waterfall.image.image.shape[1] <= 256
        for mode, selected, visible in (
            ("Single", (frames[-1],), (traces[-1],)),
            ("Overlay", frames, traces), ("Waterfall", frames, traces)):
            state = replace(delta, plot_mode=mode, traces=visible, live_update=False)
            view.reconcile(state, FrameNavigationProjection(frames, frames[-1], selected),
                           completed=3621, total=3621, detail="Ready")
            assert view._plot_mode == mode and len(view._heavy_available) <= 2
        assert view.waterfall.image.image.shape[1] <= 256
    finally:
        view.close()


def test_replacement_race_and_cleanup_failure_retry_to_one_owner(monkeypatch, tmp_path):
    controller, loader, acquisition, browse, processed, identity = _cold_browse(
        tmp_path / "race", acquisition=True)
    key = _select_label(controller, 2)
    assert browse.publication_store.discard(2)
    owner = controller._browse_hydration_owner
    entered, release = threading.Event(), threading.Event()
    original_read = transport_module.read_frame_preview

    def held(*args, **kwargs):
        entered.set(); release.wait(timeout=10.0)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(transport_module, "read_frame_preview", held)
    original_release = loader.release_context
    injected = []

    def fail_once(context):
        if not injected:
            injected.append(context)
            return controller_module.BrowseCleanupReceipt(
                context.load_request, CleanupStatus.CLEANUP_PENDING)
        return original_release(context)

    monkeypatch.setattr(loader, "release_context", fail_once)
    try:
        assert controller.project(key) is None and entered.wait(10.0)
        with pytest.raises(RuntimeError, match="previous Browse cleanup is pending"):
            controller.begin_browse(str(processed))
        assert controller.browse_context is browse and not browse.released
        assert _wait(lambda: (controller.poll_browse(), not controller.browse_pending)[1])
        release.set()
        assert _wait(lambda: owner.transport.queued_token is None and
                     owner.transport.active_token is None)
        with pytest.raises(RuntimeError, match="previous Browse cleanup is pending"):
            controller.begin_browse(str(processed))
        assert _wait(lambda: (controller.poll_browse(), not controller.browse_pending)[1])
        assert controller.poll_browse_preview() is True
        with pytest.raises(RuntimeError, match="previous Browse cleanup is pending"):
            controller.begin_browse(str(processed))
        assert injected == [browse] and controller.browse_context is browse
        assert _wait(lambda: (controller.poll_browse(), not controller.browse_pending)[1])
        request = controller.begin_browse(str(processed))

        def replaced():
            outcome = controller.poll_browse()
            return outcome is not None and outcome.request is request

        assert _wait(replaced, 20.0)
        replacement = controller.browse_context
        assert replacement is not browse and browse.released
        assert controller._browse_hydration_owner is not owner
        assert controller.selection.names(replacement)
    finally:
        release.set()
        for _ in range(20):
            receipt = controller.close()
            if receipt.cleanup_status is CleanupStatus.CLEANED:
                break
            time.sleep(0.01)
        if identity is not None:
            controller.release_acquisition(identity)
        acquisition.publication_store.transport.retire(join_timeout=1.0)


def test_workspace_close_and_exit_leave_no_legacy_viewer_cache_or_thread_owner(
    mount, monkeypatch, tmp_path,
):
    borrowed = mount.acquisition.publication_store.transport
    retire_calls = []
    original_retire = borrowed.retire
    monkeypatch.setattr(borrowed, "retire", lambda **kwargs: retire_calls.append(kwargs) or True)
    mount.open1()
    assert mount.clear1() and retire_calls == []
    monkeypatch.setattr(borrowed, "retire", original_retire)

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "host-settings.ini"))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "host-session.json"))
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _gui_main.Main(page_descriptors=(SCATTERING_WORKSPACE_PAGE,),
                            selected_page_key=SCATTERING_WORKSPACE_PAGE.key)
    page = window.main_widget
    controller = page._context_controller
    root = tmp_path / "host-xye"; root.mkdir()
    paths = tuple(str(p2c._write_xye(root / f"{index}.xye", [0, 1], [index, index + 1]))
                  for index in (1, 2))
    terminated = []
    monkeypatch.setattr(window, "_terminate_process", lambda: terminated.append(True))
    try:
        controller.open_viewer_1d(paths)

        def ready():
            controller.poll_viewer_1d()
            context = controller.viewer_1d_context
            return context is not None and context.state.value == "ready"

        assert _wait(ready)
        assert page._clear_viewer_1d_renderer(close=True)
        image = tmp_path / "host-image.npy"
        np.save(image, np.arange(12.0).reshape(3, 4))
        controller.open_viewer_2d(str(image))

        def ready_2d():
            page._drain_executor()
            return controller.viewer_2d_frame is not None

        assert _wait(ready_2d)
        standalone = controller._viewer_2d_standalone
        assert standalone is not None
        window.ui.actionExit.trigger()
        assert _wait(lambda: bool(terminated) and not window.isVisible(), 5.0)
        assert window.page_handle.close().status is PageCleanup.CLEAN
        assert controller.viewer_1d_context is controller.viewer_2d_context is None
        assert controller._viewer_2d_standalone is None and controller.selection is None
        assert controller.retained_contexts == ()
        assert not any(thread.is_alive() and thread.name.startswith("scattering-")
                       for thread in threading.enumerate())
        legacy = ("xdart.gui.tabs.static_scan.h5viewer",
                  "xdart.gui.tabs.static_scan.static_scan_widget",
                  "xdart.gui.tabs.static_scan.viewer_raw_lru",
                  "xdart.gui.tabs.static_scan.display_frame_widget")
        assert not any(type(obj).__module__.startswith(legacy)
                       for obj in window.findChildren(QtCore.QObject))
    finally:
        window._process_exit_requested = False
        window.close(); window.deleteLater(); qapp.processEvents()
