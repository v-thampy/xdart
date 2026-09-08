"""Saved numerical cuts and bounded Browse slice worker ownership."""

from dataclasses import replace
from threading import Event, get_ident
from time import monotonic, sleep
import os
import weakref

import h5py
import numpy as np
import pandas as pd
import pytest

from xdart.gui.tabs.scattering.browse_slice_hydration import BrowseSliceLane
from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection, SlicePin
from xdart.modules.display_context import BrowseContext, DisplaySelection
from xdart.modules.frame_publication import PublicationStore
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io import Browse1DCache, FrameViewReader
from xrd_tools.io.nexus import write_integrated_stack, write_scan_metadata
from xrd_tools.io.nexus_record import ensure_frames_container, write_frame_record
from xrd_tools.io.output_transaction import (
    StreamTerminal, capture_target_snapshot, revalidate_stream_terminal,
)


def _wait(call, timeout=5.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        value = call()
        if value:
            return value
        sleep(0.001)
    raise AssertionError("Browse slice worker did not settle")


@pytest.fixture
def saved(tmp_path, request):
    """Real numbered-axis output and its admitted immutable catalog."""
    path = tmp_path / "scan.nexus"
    labels = (2, 5, 9)
    q, chi = np.array([0.1, 0.2, 0.3]), np.array([-2.0, 0.0, 2.0])
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["ssrl_schema"] = "xrd_tools.processed_scan"
        entry.attrs["ssrl_schema_version"] = 3
        write_integrated_stack(
            entry, frame_indices=labels,
            results_1d=[IntegrationResult1D(q, np.array([100., 200., 300.]) + n,
                                           unit="q_A^-1") for n in labels],
            results_2d=[IntegrationResult2D(
                q, chi, np.array([[0., 1., 2.], [10., 11., 12.], [20., 21., np.nan]]) + n,
                unit="q_A^-1", azimuthal_unit="chi_deg",
            ) for n in labels] if getattr(request, "param", True) else None,
        )
        for n in labels:
            write_frame_record(ensure_frames_container(entry), f"frame_{n:04d}")
        write_scan_metadata(entry, pd.DataFrame(
            {"I0": [float(n) for n in labels], "epoch": [float(n + 10) for n in labels]},
            index=labels,
        ), labels)
    with FrameViewReader(path, include_thumbnail=False, resolve_source=False) as reader:
        catalog = reader.read_scalar_catalog()
    snapshot = capture_target_snapshot(path)
    terminal = None
    if getattr(request, "param", None) == "terminal":
        stat = path.stat()
        terminal = StreamTerminal(str(path.resolve()), stat.st_size, snapshot.digest,
                                  len(labels), stat.st_dev, stat.st_ino,
                                  stat.st_mtime_ns, stat.st_ctime_ns)
        snapshot = revalidate_stream_terminal(path, terminal)
    request = BrowseLoadRequest("browse-slice", 1, str(path.resolve()), terminal)
    cache = Browse1DCache(1 << 20)
    context = BrowseContext(
        context_token=request.token, load_generation=request.load_generation,
        operation=request, requested_path=request.source_path, scan_key="scan",
        scan=object(), frame=None, frame_ids=catalog.labels, frames={},
        viewer_rows_1d={}, viewer_rows_2d={}, publication_store=PublicationStore(),
        record_store={}, scalar_catalog=catalog, browse_1d_cache=cache,
        target_entry=catalog.entry, loaded_labels=catalog.labels,
        target_snapshot=snapshot,
        calibration_identity='{"wavelength_m": 1e-10}',
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    identity = RunIdentity(1, "browse-slice")
    frames = tuple(DisplayFrameKey(identity, "scan", request.source_path, n, i)
                   for i, n in enumerate(labels, 1))
    selection = DisplaySelection.for_context(context, 1)
    lane = BrowseSliceLane(context)
    yield path, context, frames, selection, lane
    _wait(lane.release)
    cache.close()


def _project(saved, preferences, *, selected=None, norm=""):
    _, _, frames, selection, lane = saved
    selected = frames if selected is None else selected
    navigation = FrameNavigationProjection(frames, selected[-1] if selected else None, selected)

    def completed():
        lane.consume_repaint()
        result = lane.project(selection, navigation, preferences=preferences, norm_channel=norm)
        return result if not result.pending else None

    result = _wait(completed)
    _wait(lambda: (lane.consume_repaint() or True) and not lane.polling_needed())
    return result


@pytest.mark.parametrize("mode", ["Single", "Overlay", "Waterfall"])
def test_saved_selected_cuts_use_half_width_nanmean_and_row_normalization(saved, mode):
    preferences = ScientificPreferences(plot_mode=mode, slice_enabled=True,
                                        slice_center=0.0, slice_width=0.1)
    result = _project(saved, preferences, norm="I0")
    assert result.diagnostic == "" and not result.pinned_traces
    assert tuple(trace.frame for trace in result.traces) == saved[2]
    for trace in result.traces:
        n = trace.frame.local_frame_label
        np.testing.assert_allclose(trace.intensity, (np.array([1., 11., 21.]) + n) / n)
        np.testing.assert_allclose(trace.axis.values, [.1, .2, .3])
        assert "χ=0.00±0.10" in trace.title and trace.epoch == float(n + 10)
        assert trace.axis.values.base is None and trace.intensity.base is None
        assert not trace.axis.values.flags.writeable and not trace.intensity.flags.writeable
    result = _project(saved, replace(preferences, slice_center=1.0, slice_width=1.0), norm="I0")
    for trace in result.traces:
        n = trace.frame.local_frame_label
        np.testing.assert_allclose(trace.intensity, (np.array([1.5, 11.5, 21.]) + n) / n)


def test_pin_absorption_unselected_reuse_axis_conversion_and_native_rows(saved, monkeypatch):
    frames = saved[2]
    pin = SlicePin(frames[0], "Q", 0.0, 0.1)
    preferences = ScientificPreferences(slice_enabled=True, slice_center=0.0,
                                        slice_width=0.1, slice_pins=(pin,))
    first = _project(saved, preferences)
    assert tuple(trace.frame for trace in first.traces) == frames[1:]
    assert len(first.pinned_traces) == 1
    frozen = first.pinned_traces[0].trace
    reads = []
    original = FrameViewReader.read

    def counted(reader, frame, **kwargs):
        reads.append(frame)
        return original(reader, frame, **kwargs)

    monkeypatch.setattr(FrameViewReader, "read", counted)
    moved = _project(saved, replace(preferences, slice_center=2.0), selected=(frames[2],))
    assert reads == [9]
    assert moved.pinned_traces[0].trace is frozen
    np.testing.assert_allclose(moved.traces[0].intensity, [11., 21., np.nan])
    native = _project(saved, replace(preferences, slice_enabled=False), selected=(frames[1],))
    np.testing.assert_allclose(native.traces[0].intensity, [105., 205., 305.])
    assert native.pinned_traces[0].trace is frozen
    converted = _project(saved, replace(preferences, plot_axis="2theta",
        slice_pins=(replace(pin, plot_axis="2theta"),)), selected=(frames[2],), norm="I0")
    expected_axis = np.rad2deg(2 * np.arcsin(np.array([.1, .2, .3]) / (4 * np.pi)))
    np.testing.assert_allclose(converted.pinned_traces[0].trace.axis.values, expected_axis)
    np.testing.assert_allclose(converted.pinned_traces[0].trace.intensity, [1.5, 6.5, 11.5])
    assert sorted(reads[-2:]) == [2, 9]
    _project(saved, replace(preferences, slice_pins=()), selected=(frames[2],))
    assert not saved[4]._pin_cache


def test_nonoverlap_is_empty_and_chi_cut_uses_saved_q_bins(saved):
    preferences = ScientificPreferences(slice_enabled=True, slice_center=900.0, slice_width=.1)
    empty = _project(saved, preferences)
    assert empty.traces == () and empty.diagnostic == ""
    result = _project(saved, replace(preferences, plot_axis="chi", slice_center=.2, slice_width=.01))
    for trace in result.traces:
        np.testing.assert_allclose(trace.axis.values, [-2., 0., 2.])
        np.testing.assert_allclose(trace.intensity, np.array([10., 11., 12.]) + trace.frame.local_frame_label)


def test_one_open_per_batch_and_no_cake_retained_after_projection(saved, monkeypatch):
    opened, read_threads, cakes = [], [], []
    original_enter, original_read = FrameViewReader.__enter__, FrameViewReader.read
    gui_thread = get_ident()

    def entered(reader):
        opened.append(reader)
        assert reader.include_thumbnail is False and reader.resolve_source is False
        return original_enter(reader)

    def read(reader, frame, **kwargs):
        read_threads.append(get_ident())
        view = original_read(reader, frame, **kwargs)
        assert view.raw is None and view.thumbnail is None
        cakes.append(weakref.ref(view.intensity_2d))
        return view

    monkeypatch.setattr(FrameViewReader, "__enter__", entered)
    monkeypatch.setattr(FrameViewReader, "read", read)
    result = _project(saved, ScientificPreferences(slice_enabled=True))
    assert len(result.traces) == 3 and len(opened) == 1
    assert read_threads == [read_threads[0]] * 3 and read_threads[0] != gui_thread
    assert all(ref() is None for ref in cakes)
    assert opened[0].retained_bytes == 0


def test_latest_request_supersedes_between_real_reads_and_cancel_is_reusable(saved, monkeypatch):
    frames, selection, lane = saved[2:]
    started, finish = Event(), Event()
    original = FrameViewReader.read
    reads = []

    def read(reader, frame, **kwargs):
        reads.append(frame)
        view = original(reader, frame, **kwargs)
        if len(reads) == 1:
            started.set()
            assert finish.wait(5)
        return view

    monkeypatch.setattr(FrameViewReader, "read", read)
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    preferences = ScientificPreferences(slice_enabled=True, slice_center=0.0, slice_width=.1)
    assert lane.project(selection, navigation, preferences=preferences).pending
    assert started.wait(5)
    try:
        assert lane.project(selection, navigation, preferences=replace(preferences, slice_center=-2.0)).pending
        newest = replace(preferences, slice_center=2.0)
        assert lane.project(selection, navigation, preferences=newest).pending
        assert not lane.consume_repaint()
    finally:
        finish.set()
    result = _project(saved, newest)
    assert reads == [2, 2, 5, 9]
    np.testing.assert_allclose(result.traces[0].intensity, [4., 14., np.nan])
    lane.cancel()
    assert not lane.consume_repaint() and not lane._pin_cache
    assert len(_project(saved, preferences).traces) == 3


def test_release_is_nonblocking_until_actual_reader_close_and_discards_completion(saved, monkeypatch):
    frames, selection, lane = saved[2:]
    closing, finish = Event(), Event()
    original = FrameViewReader.__exit__
    close_threads = []

    def close(reader, *args):
        close_threads.append(get_ident())
        closing.set()
        assert finish.wait(5)
        return original(reader, *args)

    monkeypatch.setattr(FrameViewReader, "__exit__", close)
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    assert lane.project(selection, navigation, preferences=ScientificPreferences(slice_enabled=True)).pending
    assert closing.wait(5)
    try:
        before = monotonic()
        assert not lane.release()
        assert monotonic() - before < .25
        assert not lane.consume_repaint()
    finally:
        finish.set()
    _wait(lane.release)
    assert not lane.polling_needed() and close_threads == [close_threads[0]]
    assert close_threads[0] != get_ident()
    refused = lane.project(selection, navigation, preferences=ScientificPreferences(slice_enabled=True))
    assert not refused.pending and "released" in refused.diagnostic


def test_changed_target_or_foreign_pin_cannot_deliver_curves(saved):
    path, context, frames, selection, lane = saved
    preferences = ScientificPreferences(slice_enabled=True)
    foreign = replace(frames[0])
    refused = _project(saved, replace(preferences, slice_pins=(SlicePin(foreign, "Q", 0., 1.),)))
    assert "owned frame" in refused.diagnostic and not refused.traces
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    refused = _project(saved, preferences)
    assert "changed" in refused.diagnostic and not refused.traces


def test_changed_target_during_read_discards_entire_batch(saved, monkeypatch):
    original = FrameViewReader.read
    path = saved[0]
    changed = False

    def read(reader, frame, **kwargs):
        nonlocal changed
        view = original(reader, frame, **kwargs)
        if not changed:
            changed = True
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return view

    monkeypatch.setattr(FrameViewReader, "read", read)
    result = _project(saved, ScientificPreferences(slice_enabled=True))
    assert "changed" in result.diagnostic and not result.traces


@pytest.mark.parametrize("saved", [True, "terminal"], indirect=True)
def test_display_requalification_never_rehashes_loaded_artifact(saved, monkeypatch):
    import xrd_tools.io.output_transaction as transaction

    def forbidden(*args, **kwargs):
        raise AssertionError("a slice edit must not hash the whole saved file")

    monkeypatch.setattr(transaction, "_capture_target", forbidden)
    preferences = ScientificPreferences(slice_enabled=True, slice_center=0.0, slice_width=.1)
    assert len(_project(saved, preferences).traces) == 3
    assert len(_project(saved, replace(preferences, slice_center=2.0)).traces) == 3


def test_failed_close_retries_same_reader_without_repeating_reads(saved, monkeypatch):
    original = FrameViewReader.__exit__
    readers = []

    def close(reader, *args):
        readers.append(reader)
        if len(readers) == 1:
            raise OSError("injected close failure")
        return original(reader, *args)

    monkeypatch.setattr(FrameViewReader, "__exit__", close)
    result = _project(saved, ScientificPreferences(slice_enabled=True))
    assert "injected close failure" in result.diagnostic
    assert not result.traces
    assert len(readers) == 2 and readers[0] is readers[1]
    assert readers[0].retained_bytes == 0


def test_preserve_pending_repaint_holds_release_until_consumed(saved):
    frames, selection, lane = saved[2:]
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    assert lane.project(selection, navigation, preferences=ScientificPreferences(slice_enabled=True)).pending
    _wait(lambda: lane._repaint)
    assert not lane.release(preserve_pending_repaint=True)
    assert lane.consume_repaint()
    _wait(lambda: lane.release(preserve_pending_repaint=True))


def test_context_invalidation_discards_inflight_completion(saved, monkeypatch):
    context, frames, selection, lane = saved[1:]
    started, finish = Event(), Event()
    original = FrameViewReader.read
    reads = []

    def read(reader, frame, **kwargs):
        reads.append(frame)
        view = original(reader, frame, **kwargs)
        started.set()
        assert finish.wait(5)
        return view

    monkeypatch.setattr(FrameViewReader, "read", read)
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    assert lane.project(selection, navigation, preferences=ScientificPreferences(slice_enabled=True)).pending
    assert started.wait(5)
    try:
        context.invalidate()
    finally:
        finish.set()
    _wait(lambda: not lane.polling_needed())
    assert not lane.consume_repaint() and reads == [2]
    result = lane.project(selection, navigation, preferences=ScientificPreferences(slice_enabled=True))
    assert not result.pending and not result.traces and "not current" in result.diagnostic


@pytest.mark.parametrize("saved", [False], indirect=True)
def test_no_saved_cake_never_uses_native_1d(saved):
    result = _project(saved, ScientificPreferences(slice_enabled=True))
    assert "no saved 2-D data" in result.diagnostic and not result.traces
