from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import xdart.gui.tabs.scattering.hydration_transport as hydration
from xdart.gui.tabs.scattering.batch_terminal_presentation import (
    BatchTerminalPresentationController,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.metadata_operations import MetadataOperationOwner
from xdart.gui.tabs.scattering.processed_browser import ProcessedBrowserOwner
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences, build_scientific_projection
from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions, ShellCommand, ShellCommandKind, SlicePin
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_operations import WorkspaceOperationOwner
from xdart.modules.display_context import ContextKind, Viewer2DRendererClearReceipt
from xrd_tools.session.hydration import HydrationCompletion, HydrationOutcome
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session.scan_norm import ScanNormAggregate
from xrd_tools.session.viewer_1d import _new_viewer_1d_renderer_clear_receipt

from tests.xdart.scattering.test_e3_context_contract import _acquisition, _browse, _cold_controller
from tests.xdart.scattering.test_p2b1_viewer_context_page import _shell as _viewer_shell, _write_xye

_ROOT = Path(__file__).parents[3]


class _Renderer:
    def __init__(self, events):
        self.events, self.ok1, self.ok2, self.workspace = events, True, True, False

    def clear_viewer_1d(self, request):
        self.events.append("clear1")
        return _new_viewer_1d_renderer_clear_receipt(request, self.ok1)

    def clear_viewer_2d(self, request):
        self.events.append("clear2")
        return Viewer2DRendererClearReceipt(request, self.ok2)

    def clear_workspace(self):
        self.events.append("workspace")
        return self.workspace

    def drop_viewer_loading_snapshot(self):
        pass


class _Mount:
    def __init__(self, tmp_path, monkeypatch):
        left, right = tmp_path / "left", tmp_path / "right"
        left.mkdir(); right.mkdir()
        self.paths = (str(_write_xye(left / "same.xye", [0, 1, 2], [2, 3, 4], [.1, .2, .3])),
                      str(_write_xye(right / "same.xye", [0, 2, 4], [5, 6, 7], [.4, .5, .6])))
        self.image = tmp_path / "two.npy"
        np.save(self.image, np.arange(24.0).reshape(2, 3, 4))
        self.controller, self.lifecycle, self.loader = _cold_controller()
        self.identity, self.acquisition = _acquisition()
        self.controller._runtime.adopt_acquisition(self.identity, self.acquisition)
        self.events, self.io, self.done1, self.done2 = [], Counter(), Event(), Event()
        for name in ("catalog_viewer_2d", "read_viewer_2d_frame", "begin_viewer_1d_read"):
            original = getattr(hydration, name)
            def counted(*args, _name=name, _original=original, **kwargs):
                self.io[_name] += 1; self.events.append(_name); return _original(*args, **kwargs)
            monkeypatch.setattr(hydration, name, counted)
        for owner, done in ((self.controller._viewer_1d, self.done1),
                            (self.controller._viewer_2d, self.done2)):
            original = type(owner).complete
            def completed(target, value, _original=original, _done=done):
                _original(target, value); _done.set()
            monkeypatch.setattr(type(owner), "complete", completed)
        begin = self.loader.begin
        self.loader.begin = lambda request: (self.events.append("browse"), begin(request))[1]
        self.renderer = _Renderer(self.events)
        self.choice2 = str(self.image)
        self.processed_browser = ProcessedBrowserOwner(
            save_path="",
            processing_mode="Int 2D",
            deliver=lambda _wake: None,
            catalog_reader=lambda _directory, **_kwargs: (),
        )
        self.page = SimpleNamespace(
            _closing=False, _closed=False, _context_controller=self.controller,
            _batch_terminal=BatchTerminalPresentationController(),
            _processed_browser=self.processed_browser,
            _metadata_operations=MetadataOperationOwner(),
            _workspace_operations=WorkspaceOperationOwner(),
            _browse_1d_release_debt=None,
            _intents=RunIntentStore(RunIntent(processing_mode="Int 2D")),
            _analysis_operation_busy=lambda: False,
            _experiment_operation_busy=lambda: False,
            _shell=SimpleNamespace(scientific=self.renderer, browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: self.events.append("cancel-browser"))),
            _last_scientific_projection=None, _presentation_targets=[],
            _viewer_2d_file_chooser=lambda _start: self.choice2,
            _viewer_1d_file_chooser=lambda _start: self.paths,
            _notice=lambda value: self.events.append(("notice", value)),
            _error_notice=lambda *value: self.events.append(("error", value)),
            _ensure_timer=lambda: self.events.append("timer"),
            _refresh_shell=lambda **_kwargs: self.events.append("refresh"))
        for name in ("_viewer_1d_start_directory", "_viewer_2d_start_directory",
                     "_clear_viewer_1d_renderer", "_clear_viewer_2d_renderer",
                     "_apply_batch_retirement", "_retire_batch_presentation",
                     "_release_browse_1d_debt",
                     "_abandon_reintegrate_display_owner", "_abandon_reintegrate_successor",
                     "_open_viewer_1d_paths", "_open_viewer_2d_path",
                     "_select_scan"):
            setattr(self.page, name, partial(getattr(ScatteringWorkspace, name), self.page))

    def wait(self, dimension):
        done, poll = ((self.done1, self.controller.poll_viewer_1d) if dimension == "1d"
                      else (self.done2, self.controller.poll_viewer_2d))
        for _ in range(8):
            poll()
            context = getattr(self.controller, f"viewer_{dimension}_context")
            frame = True if dimension == "1d" else self.controller.viewer_2d_frame is not None
            if context is not None and context.state.value == "ready" and frame:
                return
            assert done.wait(4), f"{dimension} completion did not arrive"
            done.clear()
        pytest.fail(f"{dimension} did not become ready")

    def open2(self):
        self.done2.clear(); self.controller.open_viewer_2d(str(self.image)); self.wait("2d")

    def open1(self):
        self.done1.clear(); self.controller.open_viewer_1d(self.paths); self.wait("1d")

    def browse(self, path="/processed/browse.b.nxs"):
        request = self.controller.begin_browse(path)
        _, context = _browse(request.token, request.load_generation, request=request)
        self.loader.complete(context); assert self.controller.poll_browse() is not None
        return context

    def dispatch(self, value):
        ScatteringWorkspace._handle_shell_command(
            self.page, ShellCommand(ShellCommandKind.SELECT_SCAN, value))

    def native(self, preferences, mode="Int 2D"):
        payloads = self.controller.project_navigation(preferences=preferences, processing_mode=mode)
        return build_scientific_projection(payloads, self.controller.navigation,
            self.controller.resident_frame_keys, preferences, "", RunPhase.IDLE,
            processing_mode=mode, norm_aggregate=self.controller.norm_aggregate)

    def clear1(self):
        for _ in range(8):
            if self.page._clear_viewer_1d_renderer(close=True): return True
            if not self.controller.poll_viewer_1d(): self.done1.wait(4)
            self.done1.clear()
        return False

    def finish(self):
        self.renderer.ok1 = self.renderer.ok2 = True
        for clear in (self.page._clear_viewer_1d_renderer, self.page._clear_viewer_2d_renderer):
            try: clear(close=True)
            except Exception: pass
        try: self.controller.close()
        except Exception: pass
        self.processed_browser.begin_close()
        self.processed_browser.retry_close()
        self.controller.release_acquisition(self.identity)
        self.acquisition.publication_store.transport.retire(join_timeout=1.0)


@pytest.fixture
def mount(tmp_path, monkeypatch):
    value = _Mount(tmp_path, monkeypatch)
    try: yield value
    finally: value.finish()


def _selected(mount, kind):
    selection = mount.controller.selection
    assert selection is not None and selection.kind is kind
    named = [context for context in mount.controller.retained_contexts if selection.names(context)]
    assert len(named) == 1
    return named[0]


def _viewer2_fingerprint(mount):
    owner, selection = mount.controller._viewer_2d, mount.controller.selection
    payload = mount.controller.project(mount.controller.navigation.current)
    return (selection.kind, selection.context_token, selection.source_path, owner.context.original_path,
            owner.catalog, owner.frame, mount.controller.navigation.current, payload.title, payload.status,
            payload.view.raw, owner.provider)


def _acquisition_fingerprint(mount):
    value, config, store = mount.acquisition, mount.acquisition.run_configuration, mount.acquisition.publication_store
    return (value, config, config.fingerprint, config.thaw_source_spec(), config.output_mode, config.live_mode, config.save_path,
            value.context_token, value.run_scan_key, value.source_path, value.scan, value.frame,
            value.frame_ids, value.frame_ids.snapshot(), value.frames, store, store.transport, store.catalog_snapshot(),
            tuple(store.artifacts.items()), value.poni_identity, value.origin, mount.identity)


def test_common_mount_census_and_selected_context_precedence(mount) -> None:
    expected = {"context_runtime.py": ("_ContextRuntime",),
                "context_controller.py": ("_OneDViewerOwner", "_TwoDViewerOwner", "ContextController"),
                "page.py": ("ScatteringWorkspace",)}
    for module, names in expected.items():
        tree = ast.parse((_ROOT / "src/xdart/gui/tabs/scattering" / module).read_text())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert all(classes.count(name) == 1 for name in names)
    controller = mount.controller
    assert controller._viewer_1d_lock is controller._viewer_2d_lock
    assert _selected(mount, ContextKind.ACQUISITION) is mount.acquisition
    browse = mount.browse(); assert _selected(mount, ContextKind.BROWSE) is browse
    mount.open2(); provider = controller._viewer_2d.provider
    assert _selected(mount, ContextKind.VIEWER_2D) is controller.viewer_2d_context
    assert provider is mount.acquisition.publication_store and controller._viewer_2d_standalone is None
    with pytest.raises(RuntimeError, match="renderer clear"): controller.open_viewer_1d(mount.paths)
    assert mount.page._clear_viewer_2d_renderer(close=True); mount.open1()
    assert _selected(mount, ContextKind.VIEWER_1D) is controller.viewer_1d_context
    assert controller._viewer_1d.provider is provider.transport and controller._viewer_2d_standalone is None


def test_page_routes_native_browse_viewer2d_viewer1d_transitions(mount) -> None:
    original, counts = mount.controller.selection, mount.io.copy()
    mount.choice2 = None; ScatteringWorkspace._choose_viewer_2d_file(mount.page, reload=False)
    assert mount.controller.selection is original and mount.io == counts
    mount.choice2 = str(mount.image); mount.done2.clear()
    ScatteringWorkspace._choose_viewer_2d_file(mount.page, reload=False); mount.wait("2d")
    old = _viewer2_fingerprint(mount); mount.renderer.ok2 = False; mount.done1.clear()
    ScatteringWorkspace._choose_viewer_1d_files(mount.page)
    owner = mount.controller._viewer_2d
    assert (owner.context.context_token, owner.context.original_path, owner.frame, owner.provider) == (old[1], old[3], old[5], old[10])
    assert mount.controller.viewer_1d_context is None
    mount.renderer.ok2 = True; ScatteringWorkspace._choose_viewer_1d_files(mount.page); mount.wait("1d")
    assert mount.paths[0].endswith("same.xye") and mount.paths[1].endswith("same.xye")
    assert mount.events.index("clear2") < mount.events.index("begin_viewer_1d_read")
    before = mount.controller.viewer_1d_context; holder = mount.controller._viewer_1d.holder
    mount.renderer.ok1 = False; mount.dispatch("/processed/route.nxs")
    assert mount.controller.viewer_1d_context.context_token == before.context_token
    assert mount.controller._viewer_1d.holder is holder and mount.loader.request is None
    mount.renderer.ok1 = True; mount.dispatch("/processed/route.nxs")
    request = mount.loader.request; _, browse = _browse(request.token, request.load_generation, request=request)
    mount.loader.complete(browse); assert mount.controller.poll_browse() is not None
    assert mount.events.index("clear1") < mount.events.index("browse")
    mount.open2(); mount.renderer.ok2 = True; start = len(mount.events); mount.dispatch("/out/a.nxs")
    assert _selected(mount, ContextKind.ACQUISITION) is mount.acquisition
    assert not mount.controller.viewer_2d_owned and "clear2" in mount.events[start:]


def test_native_projection_and_preferences_survive_viewer_round_trip(mount, monkeypatch) -> None:
    frame = mount.controller.navigation.current
    identity = (mount.identity.generation, mount.identity.fingerprint, str(frame.artifact), frame.source_scan)
    aggregate = ScanNormAggregate(identity, 3, 2, {"i0": (4.0, 2)})
    monkeypatch.setattr(type(mount.acquisition.publication_store), "frame_norm_aggregate", lambda _store, _key: aggregate)
    preferences = ScientificPreferences(norm_channel="i0", color_map="magma", log_scale=True,
        image_axis="2Th-Chi", plot_axis="Chi", plot_mode="Overlay", share_axis=True,
        slice_enabled=True, slice_center=1.0, slice_width=2.0,
        slice_pins=(SlicePin(frame, "Chi", 1.0, 2.0),), q_range=(1.0, 3.0),
        chi_range=(-9.0, 9.0), plot_options=ScientificPlotOptions(overlay_offset=7.0))
    native = mount.native(preferences)
    fields = ("processing_mode", "norm_channel", "norm_identity", "norm_revision", "color_map",
              "log_scale", "image_axis", "plot_axis", "plot_mode", "share_axis", "slice_enabled",
              "slice_center", "slice_width", "slice_pins", "q_range", "chi_range", "plot_options")
    saved = tuple(getattr(native, name) for name in fields)
    mount.open2(); viewer2 = _viewer_shell(mount.controller, preferences, "2D Viewer").scientific
    assert viewer2.plot_mode == "Single" and viewer2.heavy.raw is mount.controller.viewer_2d_frame.array
    assert not viewer2.traces and not viewer2.slice_pins and viewer2.norm_identity is None
    assert mount.page._clear_viewer_2d_renderer(close=True); mount.open1()
    reads = mount.io["begin_viewer_1d_read"]
    frames = mount.controller.navigation.frames
    for mode, count in (("Single", 1), ("Overlay", 2), ("Waterfall", 2)):
        assert mount.controller.select_viewer_1d(
            frames[0], (frames[0],) if mode == "Single" else frames)
        projected = _viewer_shell(mount.controller, replace(preferences, plot_mode=mode)).scientific
        assert projected.plot_mode == mode and len(projected.traces) == count
        assert projected.heavy is None and not projected.slice_pins and projected.norm_identity is None
        assert not projected.share_axis and all(trace.sigma is not None for trace in projected.traces)
    assert mount.io["begin_viewer_1d_read"] == reads; del projected
    assert mount.clear1()
    mount.controller.select_acquisition(); restored = mount.native(preferences)
    assert tuple(getattr(restored, name) for name in fields) == saved


def test_reload_zero_io_and_foreign_late_completion_are_contextual(mount) -> None:
    mount.open2(); controller, owner = mount.controller, mount.controller._viewer_2d
    old_token, counts = owner.request_token, mount.io.copy()
    assert controller.select_viewer_2d_frame(controller.navigation.current.local_frame_label)
    _viewer_shell(controller, ScientificPreferences(), "2D Viewer")
    assert mount.io == counts
    generation = controller._runtime._display_generation; mount.done2.clear()
    ScatteringWorkspace._choose_viewer_2d_file(mount.page, reload=True); mount.wait("2d")
    assert mount.io["catalog_viewer_2d"] == counts["catalog_viewer_2d"] + 1
    assert mount.io["read_viewer_2d_frame"] == counts["read_viewer_2d_frame"] + 1
    assert controller._runtime._display_generation > generation
    payload = controller.project(controller.navigation.current)
    state = (owner.context, owner.catalog, owner.frame, owner.receipt, owner.request, owner.request_token,
             owner.loading, owner.diagnostic, controller.selection, controller.navigation,
             controller._runtime._display_generation, payload.title, payload.status, payload.view.raw)
    owner.complete(HydrationCompletion(old_token, HydrationOutcome.FAILED, "foreign late"))
    payload = controller.project(controller.navigation.current)
    assert state == (owner.context, owner.catalog, owner.frame, owner.receipt, owner.request, owner.request_token,
        owner.loading, owner.diagnostic, controller.selection, controller.navigation,
        controller._runtime._display_generation, payload.title, payload.status, payload.view.raw)


def test_failure_scrub_and_projection_provenance_are_atomic(mount) -> None:
    mount.open2(); controller = mount.controller
    old = _viewer2_fingerprint(mount); old_frame = controller.navigation.current
    old_request = controller.project_request(old_frame); old_navigation = controller.navigation
    mount.renderer.ok2 = False; ScatteringWorkspace._choose_viewer_1d_files(mount.page)
    owner = controller._viewer_2d
    assert (owner.context.context_token, owner.context.original_path, owner.catalog, owner.frame,
            controller.navigation.current, owner.provider) == (old[1], old[3], old[4], old[5], old[6], old[10])
    assert controller.viewer_1d_context is None
    mount.renderer.ok2 = True; ScatteringWorkspace._choose_viewer_1d_files(mount.page); mount.wait("1d")
    scientific = _viewer_shell(controller, ScientificPreferences(plot_mode="Overlay")).scientific
    payloads = controller.project_navigation(preferences=ScientificPreferences(), processing_mode="1D Viewer")
    assert controller.selection.kind is ContextKind.VIEWER_1D and controller.navigation is not old_navigation
    assert all(payload.frame_key is not old_frame and payload.view.raw is None for payload in payloads)
    assert [tuple(payload.view.axis_1d.values) for payload in payloads] == [(0.0, 1.0, 2.0), (0.0, 2.0, 4.0)]
    assert [payload.view.axis_1d.unit for payload in payloads] == ["", ""]
    assert all(payload.view.sigma_1d is not None for payload in payloads)
    assert scientific.heavy is None and len(scientific.traces) == 2 and scientific.norm_identity is None
    assert not controller.owns_frame(old_frame) and controller.resolve_projection(old_request) is None
    with pytest.raises(RuntimeError): controller.project_request(old_frame)


def test_workspace_close_and_viewer_only_isolation_remain_truthful(mount) -> None:
    saved = _acquisition_fingerprint(mount); mount.open1()
    page = mount.page; page._terminal_close = None; page._closing = True; page._closed = False
    page._close_identity = None; page._lifecycle = mount.lifecycle
    context, holder, provider = mount.controller.viewer_1d_context, mount.controller._viewer_1d.holder, mount.controller._viewer_1d.provider
    mount.renderer.ok1 = False; mount.renderer.workspace = True
    pending = ScatteringWorkspace.close_workspace(page)
    assert pending is not None and pending.cleanup_status.value == "cleanup_pending"
    assert mount.controller.viewer_1d_context.context_token == context.context_token
    assert mount.controller._viewer_1d.holder is holder and mount.controller._viewer_1d.provider is provider
    mount.renderer.ok1 = True; assert mount.clear1(); mount.open2()
    context, frame, provider = mount.controller.viewer_2d_context, mount.controller.viewer_2d_frame, mount.controller._viewer_2d.provider
    mount.renderer.ok2 = False
    pending = ScatteringWorkspace.close_workspace(page)
    assert pending is not None and pending.cleanup_status.value == "cleanup_pending"
    assert mount.controller.viewer_2d_context.context_token == context.context_token
    assert mount.controller.viewer_2d_frame is frame and mount.controller._viewer_2d.provider is provider
    mount.renderer.ok2 = True; assert page._clear_viewer_2d_renderer(close=True)
    mount.controller.select_acquisition()
    assert _acquisition_fingerprint(mount) == saved and mount.controller._viewer_2d_standalone is None
