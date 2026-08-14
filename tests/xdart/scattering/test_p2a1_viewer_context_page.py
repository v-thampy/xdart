from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace
import time

import numpy as np
import pytest
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardEventKind, StandardRunEvent
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.display_context import (
    ContextKind, Viewer2DCleanupState, Viewer2DRendererClearReceipt,
    Viewer2DRendererClearRequest, Viewer2DReceiptPhase, Viewer2DState)
from xrd_tools.session.hydration import HydrationCompletion, HydrationOutcome, HydrationScope, HydrationToken
from xrd_tools.io import viewer_2d as viewer_api

from tests.core import test_viewer_2d as a0
from tests.xdart.scattering.test_e3_context_contract import _acquisition


_PARENT = "ddf266ec6c6f17edbf23bfabb5435b55fe772a6e"
def _ast_facts(source: str):
    tree, aliases = ast.parse(source), {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name.rsplit(".", 1)[-1]
    calls, identifiers = Counter(), Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls[aliases.get(node.func.id, node.func.id)] += 1
            elif isinstance(node.func, ast.Attribute):
                calls[node.func.attr] += 1
        if isinstance(node, ast.Name):
            identifiers[aliases.get(node.id, node.id)] += 1
        elif isinstance(node, ast.Attribute):
            identifiers[node.attr] += 1
    return tree, calls, identifiers

def _controller() -> ContextController:
    return ContextController(
        lifecycle=SimpleNamespace(phase=RunPhase.IDLE, reset_permitted=False, active_run_identity=None, attempt_run_identity=None),
        executor=None, browse_loader=SimpleNamespace(), projection=ContextProjection())

def _await_viewer(controller: ContextController, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        controller.poll_viewer_2d()
        if not controller.viewer_2d_loading and controller.viewer_2d_frame is not None:
            break
        time.sleep(0.005)
    assert not controller.viewer_2d_loading and controller.viewer_2d_frame is not None

def _clear(controller: ContextController) -> None:
    request = controller.begin_viewer_2d_renderer_clear()
    assert request is not None
    state = (controller.viewer_2d_context, controller.viewer_2d_frame, controller._viewer_2d.receipt, controller._viewer_2d.clear_request)
    for forged in (replace(request), request):
        assert not controller.acknowledge_viewer_2d_renderer_clear(
            Viewer2DRendererClearReceipt(forged, forged is not request))
        assert state == (controller.viewer_2d_context, controller.viewer_2d_frame, controller._viewer_2d.receipt, controller._viewer_2d.clear_request)
    assert controller.acknowledge_viewer_2d_renderer_clear(Viewer2DRendererClearReceipt(request, True))

def _mount_source(tmp_path, family):
    base = np.arange(12, dtype=np.uint16).reshape(3, 4)
    suffix = {"edf": ".edf", "tiff": ".tiff", "cbf": ".cbf", "raw": ".raw", "hdf": ".h5",
              "nexus": ".nxs", "csv": ".csv", "npy2": ".npy", "npy3": ".npy", "npz": ".npz"}.get(family, ".nxs")
    path, policy, expected = tmp_path / f"{family}{suffix}", None, (base,)
    if family in {"edf", "tiff", "cbf", "raw"}:
        a0._write_selected_source(path, "fabio" if family == "edf" else family, base)
        if family == "raw":
            policy = viewer_api.Viewer2DFormatPolicy(raw_detector_shape=base.shape, raw_dtype="uint16")
    elif family in {"hdf", "nexus"}:
        a0._write_hdf_stack(path, base[np.newaxis])
    elif family == "eiger":
        path, _, _ = a0._eiger_master(tmp_path / "eiger", [("data_000001.h5", base[np.newaxis])])
    elif family == "processed":
        h5py = pytest.importorskip("h5py")
        raw = tmp_path / "source.nxs"
        a0._write_hdf_stack(raw, base[np.newaxis])
        thumb = base.astype(np.uint8)
        with h5py.File(path, "w") as handle:
            a0._processed_source(handle, 2, raw.name, 0, "/entry/data/data")
            a0._processed_thumbnail(handle, 7, thumb, vmin=0.0, vmax=255.0)
        expected = (base, thumb.astype(float))
    elif family == "csv":
        np.savetxt(path, base, delimiter=",")
    elif family == "npy2":
        np.save(path, base)
    elif family == "npy3":
        expected = (base, base + 20)
        np.save(path, np.stack(expected))
    else:
        expected = (base, base + 20)
        a0._npz(path, [("image.npy", a0._npy_bytes(np.stack(expected)))])
    return path, policy, expected

@pytest.mark.parametrize("family", ("edf", "tiff", "cbf", "raw", "hdf", "nexus", "eiger", "processed", "csv", "npy2", "npy3", "npz"))
def test_catalog_first_controller_mounts_exact_identity_and_navigation(tmp_path, monkeypatch, family) -> None:
    path, policy, expected = _mount_source(tmp_path, family)
    if policy is not None:
        monkeypatch.setattr("xdart.gui.tabs.scattering.context_controller.Viewer2DFormatPolicy",
                            lambda: policy)
    controller = _controller()
    request = controller.open_viewer_2d(str(path))
    assert (request.path, controller.viewer_2d_loading) == (str(path), True)
    _await_viewer(controller)
    owner = controller._viewer_2d
    context = controller.viewer_2d_context
    assert (context.state, controller.selection.kind, controller.run_identity) == (Viewer2DState.READY, ContextKind.VIEWER_2D, None)
    with pytest.raises(RuntimeError, match="2D Viewer cleanup remains pending"):
        controller.begin_browse(str(path))
    assert owner.provider is controller._viewer_2d_standalone
    navigation = controller.navigation
    labels = (2, 7) if family == "processed" else tuple(range(len(expected)))
    assert tuple(frame.local_frame_label for frame in navigation.frames) == labels
    assert navigation.current is navigation.frames[0]
    assert controller.viewer_2d_context is controller._runtime._viewer_2d and controller.viewer_2d_frame is controller._runtime._viewer_2d_frame
    a0._slots_have_no_array(controller._viewer_2d.catalog)
    for index, value in enumerate(expected):
        if index:
            _clear(controller)
            assert controller.select_viewer_2d_frame(navigation.frames[index].local_frame_label)
            _await_viewer(controller)
        payload = controller.project(controller.navigation.current)
        assert (payload is not None and payload.view.raw is controller.viewer_2d_frame.array
                and np.array_equal(payload.view.raw, value))
        assert all(getattr(payload.view, name) is None for name in (
            "axis_1d", "intensity_1d", "sigma_1d", "axis_2d_x", "axis_2d_y",
            "intensity_2d", "sigma_2d", "thumbnail", "geometry"))
        ledger = viewer_api.viewer_2d_selected_ledger(owner.catalog, owner.frame.label)
        assert (owner.receipt.phase, owner.receipt.capacity, owner.receipt.reserved) == (Viewer2DReceiptPhase.FRAME_READY_A, ledger.budget, ledger.admission)
        kind = ({"processed": ("Processed raw", "Thumbnail preview"), "csv": ("CSV matrix",)}.get(
            family, ("NumPy array",) * len(expected) if family in {"npy2", "npy3", "npz"} else ("Raw detector",)))[index]
        assert payload.title == f"{path.name} · frame {controller.navigation.current.local_frame_label} · {kind}"
        assert payload.status == (f"2D Viewer · {kind}" + (" · Raw source unavailable; displaying stored thumbnail." if kind == "Thumbnail preview" else ""))
        token = controller._viewer_2d.request_token
        assert controller.select_viewer_2d_frame(controller.navigation.current.local_frame_label)
        assert controller._viewer_2d.request_token is token
    if family == "processed":
        assert not controller.owns_frame(replace(navigation.frames[0]))
    original = context
    assert not controller.close_viewer_2d() and controller.viewer_2d_frame is owner.frame
    _clear(controller)
    assert controller.viewer_2d_frame is None and controller._viewer_2d.receipt.phase is Viewer2DReceiptPhase.CATALOG_R
    assert controller.close_viewer_2d()
    assert controller.viewer_2d_context is controller._viewer_2d_standalone is None
    if family == "npy2":
        request = controller.open_viewer_2d(original.original_path)
        _await_viewer(controller)
        assert request.path == str(path) and controller.viewer_2d_context.generation > original.generation
        _clear(controller)
        assert controller.close_viewer_2d()

def test_retained_acquisition_generation_submit_order_and_owner_first(tmp_path) -> None:
    path = tmp_path / "retained.npy"
    np.save(path, np.arange(6.0).reshape(2, 3))
    controller = _controller()
    identity, acquisition = _acquisition()
    controller._runtime.adopt_acquisition(identity, acquisition)
    prior_generation = controller.selection.display_generation
    seen = []
    submit = controller._submit_viewer_2d_frame
    def observed(label, **values):
        seen.append((label, controller._viewer_2d_lock._is_owned()))
        return submit(label, **values)
    controller._submit_viewer_2d_frame = observed
    request = controller.open_viewer_2d(str(path))
    context = controller.viewer_2d_context
    assert context.generation == prior_generation + 1
    assert request.generation == request.token.presentation_generation == context.generation
    assert request.read_key.scope.context_token == context.context_token
    _await_viewer(controller)
    owner = controller._viewer_2d
    assert (owner.frame is controller._runtime._viewer_2d_frame
            and owner.receipt.phase is Viewer2DReceiptPhase.FRAME_READY_A
            and controller.viewer_2d_context.state is Viewer2DState.READY)
    assert owner.provider is acquisition.publication_store
    assert seen == [(0, False)]
    before = controller.navigation
    assert not controller.select_latest_navigation(plot_mode="Overlay")
    assert controller.navigation is before
    assert controller._viewer_2d_standalone is None
    _clear(controller)
    assert controller.close_viewer_2d()

def test_refusal_and_rapid_latest_use_one_scalar_and_truthful_loading(tmp_path, monkeypatch) -> None:
    path = tmp_path / "latest.npy"
    np.save(path, np.arange(36.0).reshape(3, 3, 4))
    controller = _controller()
    controller.open_viewer_2d(str(path))
    _await_viewer(controller)
    _clear(controller)
    owner = controller._viewer_2d
    provider = controller._viewer_2d_standalone
    real_submit, refused = provider.submit, []
    def return_foreign(request):
        refused.append(request)
        return HydrationToken(request.read_key, request.generation)
    monkeypatch.setattr(provider, "submit", return_foreign)
    assert not controller.select_viewer_2d_frame(1)
    refused_request = refused[0]
    assert owner.receipt.phase is Viewer2DReceiptPhase.CATALOG_R
    assert owner.receipt.request_token is owner.request is owner.request_token is None
    assert not owner.loading and owner.diagnostic == "2D Viewer transport refused request"
    def snapshot():
        return (owner.context, owner.catalog, owner.receipt, owner.frame, owner.request,
                owner.request_token, owner.loading, owner.diagnostic, owner.changed,
                owner.latest_label, controller.selection, controller.navigation,
                controller._runtime._display_generation)
    refused_state = snapshot()
    owner.complete(HydrationCompletion(refused_request.token, HydrationOutcome.FAILED, "late completion"))
    assert snapshot() == refused_state
    monkeypatch.setattr(provider, "submit", real_submit)
    entered, release, calls = Event(), Event(), []
    read = viewer_api.read_viewer_2d_frame
    def blocked(catalog, label, **values):
        calls.append(label)
        if label == 1:
            entered.set()
            assert release.wait(4)
        return read(catalog, label, **values)
    monkeypatch.setattr("xdart.gui.tabs.scattering.hydration_transport.read_viewer_2d_frame", blocked)
    assert controller.select_viewer_2d_frame(1)
    assert owner.request.generation == refused_request.generation + 1
    assert entered.wait(4)
    assert controller.select_viewer_2d_frame(1)
    request = owner.request
    old_scope = request.read_key.scope
    scope = HydrationScope(old_scope.context_token, "viewer-2d", "viewer-2d", old_scope.epoch + 1)
    key = a0._forge(request.read_key, scope=scope)
    for stale in (a0._forge(request, generation=request.generation + 1, token=HydrationToken(request.read_key, request.generation + 1)),
                  a0._forge(request, read_key=key, token=HydrationToken(key, request.generation))):
        owner.request, owner.request_token = stale, stale.token
        assert not owner.activate(stale).accepted
    owner.request, owner.request_token = request, request.token
    assert controller.select_viewer_2d_frame(2)
    assert controller._viewer_2d.latest_label == 2
    release.set()
    _await_viewer(controller)
    assert controller.viewer_2d_frame.label == 2 and calls == [1, 2]
    _clear(controller)
    assert controller.close_viewer_2d()

def test_viewer_authority_census_provider_order_refusal_and_cleanup(tmp_path, monkeypatch) -> None:
    root = Path(__file__).parents[3]
    module = "src/xdart/gui/tabs/scattering/"
    relative_paths = ("src/xrd_tools/session/readiness.py", *(module + name for name in (
        "context_runtime.py", "context_controller.py", "context_projection.py", "controls_projection.py", "run_mode_projection.py", "page.py", "scientific_view.py")))
    trees, calls, identifiers, baseline = [], Counter(), Counter(), Counter()
    for path in relative_paths:
        tree, path_calls, path_identifiers = _ast_facts((root / path).read_text())
        parent = subprocess.check_output(
            ("git", "-C", str(root), "show", f"{_PARENT}:{path}"), text=True)
        _, _, parent_identifiers = _ast_facts(parent)
        trees.append(tree)
        calls.update(path_calls)
        identifiers.update(path_identifiers)
        baseline.update(parent_identifiers)
    classes = tuple(node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    owners = tuple(node for node in classes if node.name == "_TwoDViewerOwner")
    assert len(owners) == calls["_TwoDViewerOwner"] == 1
    assert {node.name for node in owners[0].body if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")} == {
        "activate", "dispose", "commit", "complete"}
    assert calls["RLock"] == calls["HydrationTransport"] == 1
    authority_terms = (
        "target", "port", "provider", "generation", "worker", "thread", "timer", "queue", "scheduler", "cache", "watcher", "store", "lease", "writer",
        "output", "durability", "accounting", "calibration", "mask", "integration", "rsm",
        "descriptor", "archive", "parser", "mmap", "callback", "resource")
    identifiers.subtract(baseline)
    authority_delta = {name: count for name, count in identifiers.items()
                       if count and any(term in name.lower() for term in authority_terms)}
    assert authority_delta == {
        "_display_generation": 7, "_ensure_timer": 2, "_release_browse": 1,
        "_release_browse_for_viewer": 1, "display_generation": 5,
        "generation": 18, "output_supported": 1, "presentation_generation": 3,
        "port": 1, "provider": 27, "publication_store": 1, "release": 1,
        "_viewer_2d_provider": 1, "HydrationTransport": 2, "target": 5,
    }
    assert not {"StageLedger", "TargetLease"}.intersection(identifiers)
    path = tmp_path / "deliberately-missing.npy"
    controller, events, retries, outer = _controller(), [], [], object()
    provider = SimpleNamespace()
    def submit(request):
        events.append(("submit", controller._viewer_2d_lock._is_owned()))
        provider.request = request
        provider.activation = controller._viewer_2d.activate(request)
        return object()
    states = iter((Viewer2DCleanupState.CLEANUP_PENDING, Viewer2DCleanupState.CLEANED))
    provider.submit_viewer_2d = submit
    provider.cancel_viewer_2d = lambda _gate: events.append("cancel")
    provider.viewer_2d_blocked_cleanup_token = lambda _token: outer
    provider.retry_viewer_2d_blocked_cleanup = lambda token: retries.append(token) or SimpleNamespace(state=next(states))
    provider.viewer_2d_retains_gate = lambda _gate: False
    controller._browse_request = object()
    monkeypatch.setattr(controller, "_invalidate_browse_request", lambda: events.append("pending") or setattr(controller, "_browse_request", None))
    monkeypatch.setattr(controller, "_release_browse_for_viewer", lambda: events.append("browse"))
    monkeypatch.setattr(controller, "_viewer_2d_provider", lambda: events.append("provider") or provider)
    assert controller.open_viewer_2d(str(path)) is None
    assert events[:4] == ["pending", "browse", "provider", ("submit", False)]
    assert provider.activation.accepted and not controller._viewer_2d.activate(provider.request).accepted
    assert controller.begin_viewer_2d_renderer_clear() is None and not controller.close_viewer_2d()
    assert controller.viewer_2d_cleanup_pending and controller.close_viewer_2d()
    assert retries == [outer, outer] and controller.selection is None
    controller = _controller()
    lifecycle = controller._lifecycle
    for phase in (RunPhase.RUNNING, RunPhase.PAUSING, RunPhase.PAUSED, RunPhase.RESUMING, RunPhase.STOPPING, RunPhase.FINALIZING):
        lifecycle.phase = phase
        assert not controller._viewer_2d_admissible()
    lifecycle.phase = RunPhase.FAILED
    assert not controller._viewer_2d_admissible()
    lifecycle.reset_permitted = True
    assert controller._viewer_2d_admissible()
    for name in ("active_run_identity", "attempt_run_identity"):
        setattr(lifecycle, name, object())
        assert not controller._viewer_2d_admissible()
        setattr(lifecycle, name, None)

def test_page_clear_select_scan_and_delayed_event_are_fenced() -> None:
    calls: list[object] = []
    clear_request = object()
    identity = RunIdentity(2, "delayed")
    controller = SimpleNamespace(
        viewer_2d_owned=True, viewer_2d_frame=object(), run_identity=identity,
        poll_viewer_2d=lambda: False, poll_browse_preview=lambda: False,
        browse_pending=False, adopt_acquisition=lambda value: calls.append(("adopt", value)),
        begin_viewer_2d_renderer_clear=lambda: clear_request,
        acknowledge_viewer_2d_renderer_clear=lambda receipt: calls.append(("ack", receipt)) or receipt == "positive",
        close_viewer_2d=lambda: calls.append("close") or True,
    )
    scientific = SimpleNamespace(clear_viewer_2d=lambda _request: "forged")
    page = SimpleNamespace(
        _closing=False, _closed=False, _context_controller=controller,
        _shell=SimpleNamespace(scientific=scientific),
        _last_scientific_projection=object(),
        _select_scan=lambda value: calls.append(value), _poll_admission=lambda: False,
        _run_executor=SimpleNamespace(drain_events=lambda: (StandardRunEvent(identity, StandardEventKind.CONTEXT_READY),)),
        _lifecycle=SimpleNamespace(active_run_identity=identity, attempt_run_identity=None),
        _refresh_shell=lambda: calls.append("refresh"), _polling_needed=lambda: True,
        _run_timer=SimpleNamespace(stop=lambda: calls.append("stop")),
    )
    page._clear_viewer_2d_renderer = partial(ScatteringWorkspace._clear_viewer_2d_renderer, page)
    command = ShellCommand(ShellCommandKind.SELECT_SCAN, "/retained/result.nxs")
    ScatteringWorkspace._handle_shell_command(page, command)
    assert calls == [("ack", "forged")]
    ScatteringWorkspace._drain_executor(page)
    assert not any(isinstance(item, tuple) and item[0] == "adopt" for item in calls)
    scientific.clear_viewer_2d = lambda _request: "positive"
    ScatteringWorkspace._handle_shell_command(page, command)
    assert calls[-3:] == [("ack", "positive"), "close", "/retained/result.nxs"]
    assert page._last_scientific_projection is None

def test_real_viewer_chooser_preserves_opaque_identity_and_acquisition_isolation(monkeypatch) -> None:
    calls = []
    selected = "/opaque/data.npz::nested/image.npy"
    def forbidden(*_args, **_kwargs):
        raise AssertionError("viewer chooser entered an acquisition seam")
    for name in (
        "image_series_spec", "single_image_spec", "reduce_source_selection",
        "is_single_image_spec", "source_mode", "browse_start_dir", "remember_browse_path",
        "_typed_file_source"):
        monkeypatch.setattr(f"xdart.gui.tabs.scattering.page.{name}", forbidden)
    filesystem = SimpleNamespace(path=SimpleNamespace(**dict.fromkeys(
        ("abspath", "expanduser", "exists", "isfile", "realpath", "splitext"), forbidden)), stat=forbidden, open=forbidden)
    monkeypatch.setattr("xdart.gui.tabs.scattering.page.os", filesystem)
    controller = SimpleNamespace(
        viewer_2d_context=None, run_identity=None,
        open_viewer_2d=lambda path: calls.append(("open", path)) or object())
    intent = SimpleNamespace(processing_mode="2D Viewer", live_mode=False)
    page = SimpleNamespace(
        _context_controller=controller, _lifecycle=SimpleNamespace(phase=RunPhase.IDLE),
        _viewer_2d_start_directory=lambda: calls.append("start") or "/viewer",
        _viewer_file_chooser=lambda start: calls.append(("choose", start)) or selected,
        _clear_viewer_2d_renderer=forbidden,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(thaw=lambda: intent), commit=forbidden),
        _source_mode="directory", _source_history={"directory": object()},
        _live_source_refresh_source=object(),
        _notice=lambda value: calls.append(("notice", value)),
        _ensure_timer=lambda: calls.append("timer"), _error_notice=forbidden,
        **dict.fromkeys(("select_source", "_queue_live_source_refresh", "_begin_run"), forbidden),
    )
    page._choose_viewer_2d_file = partial(ScatteringWorkspace._choose_viewer_2d_file, page)
    source_state = (page._source_mode, dict(page._source_history), page._live_source_refresh_source)
    ScatteringWorkspace._run_action(page)
    assert calls == ["start", ("choose", "/viewer"), ("open", selected), ("notice", ""), "timer"]
    assert source_state == (page._source_mode, page._source_history, page._live_source_refresh_source)
    controller.viewer_2d_context = SimpleNamespace(original_path=selected)
    page._viewer_file_chooser = forbidden
    page._clear_viewer_2d_renderer = lambda: calls.append("clear") or True
    ScatteringWorkspace._run_action(page)
    assert calls[-4:] == ["clear", ("open", selected), ("notice", ""), "timer"]
def _set(widget, attribute, action, value) -> None:
    setattr(widget, attribute, value)
    widget.events.append((widget.name, action, value))

def _widget(value="", *, events=None, name=""):
    widget = SimpleNamespace(
        value=value, hidden=False, enabled=True,
        events=[] if events is None else events, name=name)
    widget.setText = partial(_set, widget, "value", "text")
    widget.setEnabled = partial(_set, widget, "enabled", "enabled")
    widget.setChecked = partial(_set, widget, "value", "checked")
    widget.hide = partial(_set, widget, "hidden", "hide", True)
    widget.show = partial(_set, widget, "hidden", "show", False)
    def set_visible(visible):
        widget.hidden = not visible
        widget.events.append((widget.name, "visible", visible))
    widget.setVisible = set_visible
    return widget

def _pane(*, fail=False, events=None, name="pane"):
    pane = _widget(events=events, name=name)
    pane.fail, pane.rendered, pane.render_values = fail, None, None
    histogram = SimpleNamespace(
        values=(9.0, 99.0), lo_lim=9.0, hi_lim=99.0, events=pane.events, name=f"{name}-color",
        setLevels=lambda values: _set(histogram, "values", "levels", values))
    view = SimpleNamespace(
        range=(0.0, 9.0, 0.0, 99.0), events=pane.events, name=f"{name}-range",
        setRange=lambda rect: _set(view, "range", "range", rect))
    image = SimpleNamespace(
        image=np.ones((2, 3)), levels=(9.0, 99.0),
        qimage=object(), _defferedLevels=object(), _lastDownsample=(2, 2), _displayBuffer=object(),
        _processingBuffer=object(), _imageNanLocations=object(), _imageHasNans=True,
        transform="frame-transform", events=pane.events, name=name,
        pos_label=_widget("pixel", events=pane.events, name=f"{name}-hover"),
        resetTransform=lambda: _set(image, "transform", "transform", None),
        prepareGeometryChange=lambda: pane.events.append((name, "geometry")),
        informViewBoundsChanged=lambda: pane.events.append((name, "bounds")),
        update=lambda: pane.events.append((name, "update")))
    pane.canvas = SimpleNamespace(
        imageItem=image, histogram=histogram, imageViewBox=view,
        raw_image=np.ones((2, 3)), displayed_image=np.ones((2, 3)),
        _level_cache=object(), _level_scan_token=object())
    def render(value, **_kwargs):
        pane.events.append((name, "render", value))
        if pane.fail:
            raise pane.fail if isinstance(pane.fail, BaseException) else ValueError("secret")
        pane.rendered = value
        pane.render_values = _kwargs
        pane.canvas.raw_image = pane.canvas.displayed_image = value
        image.image, image.levels, image.transform = value, (0.0, 11.0), value.shape
        histogram.values, histogram.lo_lim, histogram.hi_lim = (0.0, 11.0), 0.0, 11.0
        view.range = (0.0, float(value.shape[1]), 0.0, float(value.shape[0]))
    pane.render = render
    pane.clear = lambda: setattr(pane, "rendered", None)
    return pane

def _selector(events):
    selector = SimpleNamespace(
        items=[object()], captions=[], index=0, events=events, name="selector")
    def clear():
        selector.items.clear()
        selector.captions.clear()
        events.append(("selector", "clear"))
    def add_frame(caption, frame, _tooltip):
        selector.items.append(frame)
        selector.captions.append(caption)
    selector.clear = clear
    selector.setCurrentIndex = partial(_set, selector, "index", "index")
    selector.add_frame = add_frame
    selector.count = lambda: len(selector.items)
    selector.itemData = lambda index: selector.items[index]
    return selector

def _fake_scientific(*, failing=False):
    events = []
    raw, cake = (_pane(fail=failing, events=events, name="raw"),
                 _pane(events=events, name="cake"))
    bottom, splitter = (_widget(events=events, name="bottom"),
                        _widget(events=events, name="splitter"))
    vertical = SimpleNamespace(widget=lambda _index: bottom, sizes=[500, 500])
    vertical.setSizes = partial(setattr, vertical, "sizes")
    widgets = {name: _widget(events=events, name=name) for name in (
        "previous_frame", "next_frame", "title", "status", "progress",
        "log_scale", "norm", "background", "image_axis", "share_axis", "slice",
        "slice_center", "slice_width", "pin")}
    view = SimpleNamespace(
        events=events, raw=raw, cake=cake, image_splitter=splitter,
        vertical_splitter=vertical, frame_selector=_selector(events), color_map=object(),
        curve=_pane(events=events, name="curve"), waterfall=_pane(events=events, name="waterfall"),
        _processing_mode="", _selector_operations=0,
        _viewer_2d_payload=np.ones((2, 2)), _frame_keys=(object(),),
        _selected_keys=(object(),), _label_indices={1: [0]},
        _heavy_available=frozenset((object(),)), _trace_history_scope=object(),
        _trace_selection_keys=(object(),), _trace_history_keys=(object(),),
        _trace_history_by_identity={1: object()}, _pinned_trace_scope=object(),
        _pinned_trace_by_id={1: object()}, _rendered_trace_keys=(object(),),
        _rendered_plot_mode="Overlay", _rendered_plot_options=object(),
        _rendered_overlay_step=1.0, _bottom_waterfall_active=True,
        _waterfall_y_values=(1.0,), _waterfall_source_keys=(object(),),
        _waterfall_render_contract=object(), _rendered_image_axis="Q-Chi",
        _rendered_cake_axis_key="Q", _rendered_cake_x_axis=object(),
        _rendered_cake_y_axis=object(), _rendered_trace_axis_key="Q", **widgets,
    )
    view._set_share_link = lambda _on: None
    view._rebuild_frames = partial(ScientificView._rebuild_frames, view)
    view._apply_processing_layout = partial(ScientificView._apply_processing_layout, view)
    return view

def _assert_neutral(view, title, status) -> None:
    assert view._viewer_2d_payload is None
    assert view._frame_keys == view._selected_keys == view._rendered_trace_keys == ()
    assert view._label_indices == view._trace_history_by_identity == view._pinned_trace_by_id == {}
    assert view._heavy_available == frozenset() and view.curve.rendered is view.waterfall.rendered is None
    assert view._trace_selection_keys == view._trace_history_keys == view._waterfall_y_values == view._waterfall_source_keys == ()
    assert view._trace_history_scope is view._pinned_trace_scope is view._rendered_plot_options is None
    assert (view._rendered_plot_mode, view._bottom_waterfall_active) == ("", False)
    assert (view._rendered_overlay_step, view._waterfall_render_contract, view._rendered_image_axis,
            view._rendered_cake_axis_key, view._rendered_cake_x_axis, view._rendered_cake_y_axis,
            view._rendered_trace_axis_key) == (None,) * 7
    for pane in (view.raw, view.cake):
        assert pane.canvas.raw_image.size == pane.canvas.displayed_image.size == 0
        assert pane.canvas.imageItem.image is pane.canvas.imageItem.levels is None
        assert all(getattr(pane.canvas.imageItem, name) is None for name in (
            "qimage", "_defferedLevels", "_displayBuffer", "_processingBuffer", "_imageNanLocations", "_imageHasNans"))
        assert pane.canvas.imageItem._lastDownsample == (1, 1)
        assert pane.canvas._level_cache is pane.canvas._level_scan_token is None
        assert pane.canvas.histogram.values == (0.0, 1.0)
        assert pane.canvas.histogram.lo_lim is pane.canvas.histogram.hi_lim is None
        assert pane.canvas.imageItem.transform is None
        rect = pane.canvas.imageViewBox.range
        assert (rect.x(), rect.y(), rect.width(), rect.height()) == (0.0, 0.0, 1.0, 1.0)
    assert {"geometry", "bounds", "update"} <= {event[1] for event in view.events}
    assert view.frame_selector.items == [] and view.frame_selector.index == -1
    assert not view.previous_frame.enabled and not view.next_frame.enabled
    assert (view.title.value, view.status.value, view.progress.value) == (title, status, "0/0")
    assert all(item.hidden for item in (
        view.image_splitter, view.raw, view.cake, view.vertical_splitter.widget(1)))

def test_viewer_transaction_hides_renders_reveals_last_and_retries_same_array(monkeypatch) -> None:
    view = _fake_scientific(failing=True)
    array = np.arange(12.0).reshape(3, 4)
    identity = RunIdentity(1, "viewer-token")
    prior = DisplayFrameKey(identity, "viewer-2d", "viewer-2d", 2, 1)
    frame = DisplayFrameKey(identity, "viewer-2d", "viewer-2d", 7, 2)
    state = SimpleNamespace(
        processing_mode="2D Viewer", heavy=SimpleNamespace(frame=frame, raw=array),
        heavy_available=frozenset((frame,)), color_map="plasma", log_scale=True,
        title="image.npy · frame 7 · NumPy array", status="2D Viewer · NumPy array")
    navigation = SimpleNamespace(frames=(prior, frame), current=frame, selected=(frame,))
    monkeypatch.setattr("xdart.gui.tabs.scattering.scientific_view.QtCore.QSignalBlocker", lambda _widget: object())
    monkeypatch.setattr("xdart.gui.tabs.scattering.scientific_view.set_combo_value", lambda *_args, **_kwargs: "viridis")
    def reconcile(target=view):
        return ScientificView.reconcile(
            target, state, navigation, completed=2, total=2, detail="")
    with pytest.raises(RuntimeError, match="2D Viewer render failed") as caught:
        reconcile()
    error = caught.value
    assert (str(error), error.__cause__, error.__context__, error.__suppress_context__) == (
        "2D Viewer render failed.", None, None, True)
    _assert_neutral(view, "2D Viewer · Render failed", "2D Viewer · Render failed; retry available.")
    clear = view.raw.clear
    view.raw.clear = lambda: (_ for _ in ()).throw(ValueError("scrub"))
    with pytest.raises(RuntimeError, match="2D Viewer render failed"):
        reconcile()
    _assert_neutral(view, "2D Viewer · Render failed", "2D Viewer · Render failed; retry available.")
    view.raw.clear = clear
    view.raw.fail = False
    view.events.clear()
    reconcile()
    assert view.raw.rendered is view._viewer_2d_payload is array
    assert view.raw.render_values["level_scan_token"] == (id(frame), id(array))
    assert (view.raw.render_values["color_map"], view.raw.render_values["log_scale"]) == ("plasma", True)
    assert (view.raw.canvas.imageItem.levels, view.raw.canvas.imageItem.transform, view.raw.canvas.histogram.values,
            view.raw.canvas.histogram.lo_lim, view.raw.canvas.histogram.hi_lim, view.raw.canvas.imageViewBox.range) == (
        (0.0, 11.0), array.shape, (0.0, 11.0), 0.0, 11.0, (0.0, 4.0, 0.0, 3.0))
    assert view.progress.value == "2/2" and view.frame_selector.captions == ["2", "7"]
    assert view.cake.rendered is view.curve.rendered is view.waterfall.rendered is None
    assert view.norm.hidden and view.background.hidden
    render_at = next(i for i, event in enumerate(view.events) if event[:2] == ("raw", "render"))
    reveal_at = next(i for i, event in enumerate(view.events)
                     if event == ("splitter", "visible", True))
    assert any(event[:2] == ("splitter", "hide") for event in view.events[:render_at])
    assert any(event[:2] == ("raw", "hide") for event in view.events[:render_at])
    required = {("raw-hover", "text"), ("selector", "index"),
                ("title", "text"), ("status", "text"), ("progress", "text"),
                ("previous_frame", "enabled"), ("next_frame", "enabled")}
    positions = [max(i for i, event in enumerate(view.events) if event[:2] == key) for key in required]
    assert render_at < min(positions) and max(positions) < reveal_at
    assert view.raw.canvas.imageItem.pos_label.value == ""
    assert view.previous_frame.enabled and not view.next_frame.enabled
    forged = DisplayFrameKey(RunIdentity(9, "native"), "viewer-2d", "viewer-2d", 99, 1)
    commands = []
    view.commandRequested = SimpleNamespace(emit=commands.append)
    for selected, kind in ((navigation.current, ShellCommandKind.SELECT_FRAME),
                           (forged, ShellCommandKind.HYDRATE_FRAME),
                           (replace(navigation.current), ShellCommandKind.HYDRATE_FRAME)):
        commands.clear()
        view.frame_selector.items = [selected]
        ScientificView._frame_selected(view, 0)
        assert commands[0].kind is kind
    canonical = view._viewer_2d_payload
    request = Viewer2DRendererClearRequest("viewer-token", 3, "0" * 64, 0)
    receipt = ScientificView.clear_viewer_2d(view, request)
    assert receipt.request is request and receipt.cleared
    assert canonical is array and view.raw.rendered is None
    _assert_neutral(view, "Current", "")
    interrupt = _fake_scientific()
    interrupt.raw.fail = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        reconcile(interrupt)

def test_processing_layout_rederives_viewer_and_both_native_rows_non_gui() -> None:
    view = _fake_scientific()
    panes = (view.image_splitter, view.raw, view.cake, view.vertical_splitter.widget(1))
    ScientificView._apply_processing_layout(view, "2D Viewer")
    assert not view.image_splitter.hidden and not view.raw.hidden
    assert view.cake.hidden and view.vertical_splitter.widget(1).hidden
    ScientificView._apply_processing_layout(view, "Int 2D")
    assert not any(item.hidden for item in panes)
    assert view.vertical_splitter.sizes == [500, 500]
    ScientificView._apply_processing_layout(view, "Int 1D")
    assert view.image_splitter.hidden
    assert not any(item.hidden for item in panes[1:])
    ScientificView._apply_processing_layout(view, "Int 2D")
    assert not any(item.hidden for item in panes)
