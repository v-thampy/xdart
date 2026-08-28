from __future__ import annotations

import ast
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from xdart.gui.tabs.scattering.browser_catalog import BrowserCatalogEntry
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import RunIdentity, StandardEventKind, StandardRunEvent
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
import xdart.gui.tabs.scattering.page as page_module
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    FrameSelectionIntent,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.display_context import ContextKind
from xrd_tools.session.hydration import HydrationCompletion, HydrationOutcome
from xrd_tools.session.run_configuration import RunIntent

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_e3_context_contract import _acquisition

_ROOT = Path(__file__).parents[3]
_PRODUCTION = ("context_controller.py", "context_runtime.py", "context_projection.py",
               "controls_projection.py", "run_mode_projection.py", "page.py",
               "scientific_view.py")

def _controller() -> ContextController:
    return ContextController(lifecycle=SimpleNamespace(phase=RunPhase.IDLE, reset_permitted=False,
        active_run_identity=None, attempt_run_identity=None), executor=None,
        browse_loader=SimpleNamespace(), projection=ContextProjection())

def _api(target, name):
    value = getattr(target, name, None)
    assert value is not None, f"P2-B1 API is not mounted: {name}"
    return value

def _write_xye(path: Path, x, y, sigma=None) -> Path:
    columns = [np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
    if sigma is not None:
        columns.append(np.asarray(sigma, dtype=float))
    np.savetxt(path, np.column_stack(columns))
    return path

def _await_ready(controller: ContextController, timeout: float = 5.0) -> None:
    poll = _api(controller, "poll_viewer_1d")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        poll()
        context = getattr(controller, "viewer_1d_context", None)
        if (context is not None and context.state.value == "ready"
                and not controller.viewer_1d_loading):
            return
        time.sleep(0.005)
    context = getattr(controller, "viewer_1d_context", None)
    assert (context is not None and context.state.value == "ready"
            and not controller.viewer_1d_loading)


def _positive_clear(controller: ContextController, paths=None) -> bool:
    import xrd_tools.session.viewer_1d as viewer
    request = _api(controller, "begin_viewer_1d_renderer_clear")(paths)
    assert request is not None
    receipt = viewer._new_viewer_1d_renderer_clear_receipt(request, True)
    return bool(_api(controller, "acknowledge_viewer_1d_renderer_clear")(receipt))


def _close_1d(controller: ContextController) -> None:
    context = getattr(controller, "viewer_1d_context", None)
    if context is not None and context.state.value == "ready":
        assert _positive_clear(controller)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not _api(controller, "close_viewer_1d")():
        _api(controller, "poll_viewer_1d")()
        time.sleep(0.005)
    assert controller.close_viewer_1d()


def _shell(
    controller: ContextController,
    preferences: ScientificPreferences,
    processing_mode="1D Viewer",
    *,
    browser_catalog: tuple[BrowserCatalogEntry, ...] = (),
):
    base = make_shell_projection(plot_mode="Single")
    intent = RunIntent(processing_mode=processing_mode, output_mode="")
    payloads = controller.project_navigation(preferences=preferences, processing_mode="1D Viewer")
    context = controller.viewer_1d_context
    if not browser_catalog and context is not None:
        browser_catalog = tuple(
            BrowserCatalogEntry(path, Path(path).name, index)
            for index, path in enumerate(context.paths)
        )
    return ContextProjection().build_shell(
        revision=1, controls=base.controls, controls_readiness=base.controls_readiness,
        phase=RunPhase.IDLE, intent=intent, contexts=controller.projectable_contexts,
        selection=controller.selection, navigation=controller.navigation, payloads=payloads,
        resident_frames=controller.resident_frame_keys, progress=base.progress,
        preferences=preferences, browser_directory="",
        browser_catalog=browser_catalog,
        date_sorted=False, auto_last=True,
        viewer_1d_paths=(
            ()
            if context is None
            else context.paths
        ),
        executor_available=False, start_permitted=True, start_blocker="",
        notice=getattr(controller, "viewer_1d_diagnostic", ""),
    )


def _snapshot(controller: ContextController):
    owner = _api(controller, "_viewer_1d")
    context = controller.viewer_1d_context
    return (context, owner.request, owner.request_token, owner.holder, owner.batch_identity, owner.diagnostic, controller.selection,
            controller.navigation, controller._runtime._display_generation)


def test_viewer_1d_mount_uses_one_owner_three_method_port_and_raw_provider(tmp_path, monkeypatch) -> None:
    module = _ROOT / "src/xdart/gui/tabs/scattering/context_controller.py"
    tree = ast.parse(module.read_text())
    owners = [node for node in tree.body
              if isinstance(node, ast.ClassDef) and node.name == "_OneDViewerOwner"]
    assert len(owners) == 1
    public = {
        node.name for node in owners[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public == {"commit", "cleanup_pending", "complete"}
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.count("_OneDViewerOwner") == 1
    controller = _controller()
    assert controller._viewer_1d.controller is controller
    assert controller._viewer_1d_lock is controller._viewer_2d_lock
    path = _write_xye(tmp_path / "standalone.xye", [0, 1, 2], [4, 5, 6], [.1, .2, .3])
    request = _api(controller, "open_viewer_1d")((str(path),))
    assert request.port is controller._viewer_1d
    assert request.admitted_provider_identity is controller._viewer_2d_standalone
    assert request.commit_gate is controller.viewer_1d_context.commit_gate
    assert request.owner_identity is controller._viewer_1d.owner_identity
    assert request.owner_request_claim is controller._viewer_1d.owner_request_claim
    _await_ready(controller)
    owner = controller._viewer_1d
    assert owner.holder is not None and owner.holder.borrow is not None
    assert controller.selection.kind is ContextKind.VIEWER_1D
    assert controller.navigation.current is controller.navigation.frames[0]
    assert controller.navigation.selected == controller.navigation.frames
    assert controller.resident_frame_keys == frozenset(controller.navigation.frames)
    prior_frames = controller.navigation.frames
    prior_reads = sum(owner.provider.counters().values())
    assert _positive_clear(controller, (str(path),))
    _await_ready(controller)
    assert owner.request is not request and owner.request.token is not request.token
    assert all(new is not old for new, old in zip(controller.navigation.frames, prior_frames, strict=True))
    assert sum(owner.provider.counters().values()) == prior_reads + 1
    assert _positive_clear(controller)
    retirement, real_retire = [False], type(owner.provider).retire
    monkeypatch.setattr(type(owner.provider), "retire", lambda raw, **kw: retirement.pop() if retirement else real_retire(raw, **kw))
    assert not controller.close_viewer_1d() and controller.viewer_1d_cleanup_pending
    assert controller.close_viewer_1d()

    acquired = _controller()
    identity, acquisition = _acquisition()
    acquired._runtime.adopt_acquisition(identity, acquisition)
    raw = acquisition.publication_store.transport
    request = acquired.open_viewer_1d((str(path),))
    assert request.admitted_provider_identity is raw
    assert acquired._viewer_1d.provider is raw
    assert acquired._viewer_2d_standalone is None
    _await_ready(acquired)
    _close_1d(acquired)
    assert raw.retire(join_timeout=1.0)
    assert acquired.open_viewer_1d((str(path),)) is None
    assert acquired._viewer_1d.provider is raw
    assert acquired._viewer_2d_standalone is None
    _close_1d(acquired)


def test_viewer_1d_replacement_fences_stale_completion_and_clears_before_release(tmp_path, monkeypatch) -> None:
    import xdart.gui.tabs.scattering.context_controller as context_module
    import xrd_tools.session.viewer_1d as viewer
    paths = tuple(str(_write_xye(tmp_path / f"{name}.xye", [0, 1], values))
                  for name, values in (("a", [1, 2]), ("b", [3, 4]), ("c", [5, 6])))
    controller = _controller()
    events = []
    def spy(name, operation):
        def call(*args, **kwargs):
            events.append(name); return operation(*args, **kwargs)
        return call
    monkeypatch.setattr(type(controller._runtime), "prepare_viewer_1d_navigation", spy(
        "prepare", type(controller._runtime).prepare_viewer_1d_navigation))
    monkeypatch.setattr(context_module, "adopt_prepared_viewer_1d", spy("adopt", context_module.adopt_prepared_viewer_1d))
    monkeypatch.setattr(viewer.Viewer1DAdoptionTransfer, "owner_holder", spy(
        "holder", viewer.Viewer1DAdoptionTransfer.owner_holder))
    canonical, checks = context_module.viewer_1d_request_is_canonical, []
    monkeypatch.setattr(context_module, "viewer_1d_request_is_canonical", lambda *args: (checks.append(args), len(checks) == 1)[1])
    controller.open_viewer_1d((paths[0],)); deadline = time.monotonic() + 4
    while time.monotonic() < deadline and controller.viewer_1d_loading: controller.poll_viewer_1d(); time.sleep(0.005)
    assert len(checks) == 2 and events == [] and controller._viewer_1d.holder is None and not controller.viewer_1d_loading
    monkeypatch.setattr(context_module, "viewer_1d_request_is_canonical", canonical); _close_1d(controller)
    controller = _controller()
    first = _api(controller, "open_viewer_1d")((paths[0],))
    _await_ready(controller)
    assert events == ["prepare", "adopt", "holder"]
    active, submits = [controller], []
    provider, real_submit = controller._viewer_1d.provider, type(controller._viewer_1d.provider).submit
    def submit(raw, request):
        submits.append((request, active[0]._viewer_2d_lock._is_owned(), request.commit_gate._reserved_epoch))
        return real_submit(raw, request)
    monkeypatch.setattr(type(provider), "submit", submit)
    mints, real_request = [], ContextController._viewer_1d_request
    def mint(target, context, policy, provider):
        mints.append((context.generation, context.commit_gate._reserved_epoch, target._viewer_2d_lock._is_owned()))
        return real_request(target, context, policy, provider)
    monkeypatch.setattr(ContextController, "_viewer_1d_request", mint)
    transport_calls = []
    for name in ("cancel_gate", "retains_gate", "blocked_cleanup_token", "retry_blocked_cleanup", "retire"):
        operation = getattr(type(provider), name)
        def probe(raw, *args, _name=name, _operation=operation, **kwargs):
            transport_calls.append((_name, active[0]._viewer_2d_lock._is_owned()))
            return _operation(raw, *args, **kwargs)
        monkeypatch.setattr(type(provider), name, probe)
    counters = provider.counters()
    gate = first.commit_gate
    generation_a = first.generation
    release_events = []
    real_release = viewer.Viewer1DOwnerCleanupHolder.release
    blocked = [True]
    def release(holder, reason):
        release_events.append((reason, controller._viewer_2d_lock._is_owned()))
        if blocked:
            blocked.pop()
            if holder.borrow is not None:
                holder.borrow.close()
                holder.borrow = None
            return False
        return real_release(holder, reason)
    monkeypatch.setattr(viewer.Viewer1DOwnerCleanupHolder, "release", release)
    clear = controller.begin_viewer_1d_renderer_clear((paths[1],))
    reserved = gate._reserved_epoch
    generation_b = controller._runtime._display_generation
    assert reserved == first.read_key.scope.epoch + 1
    assert clear.display_generation == generation_b == generation_a + 1
    assert release_events == []
    assert not controller.acknowledge_viewer_1d_renderer_clear(
        viewer._new_viewer_1d_renderer_clear_receipt(clear, True)
    )
    assert release_events == [("viewer 1-D replacement", False)]
    assert controller.viewer_1d_context.state.value == "cleanup_pending"
    assert controller._viewer_1d.request is first
    assert gate._reserved_epoch == reserved
    assert controller.open_viewer_1d((paths[2],)) is None
    generation_c = controller._runtime._display_generation
    assert generation_c == generation_b + 1
    assert gate._reserved_epoch == reserved
    assert controller._viewer_1d.latest_intent.paths == (paths[2],)
    assert submits == [] and mints == [] and provider.counters() == counters
    assert controller.poll_viewer_1d()
    _await_ready(controller)
    latest = controller._viewer_1d.request
    assert submits == [(latest, False, 0)]
    assert mints == [(generation_c, 0, True)]
    assert latest.paths == (paths[2],)
    assert latest.read_key.scope.epoch == reserved
    assert latest.generation == generation_c
    assert gate._reserved_epoch == 0 and gate.epoch == reserved
    assert transport_calls == []
    before = _snapshot(controller)
    controller._viewer_1d.complete(HydrationCompletion(
        first.token, HydrationOutcome.FAILED, "late A",
    ))
    assert _snapshot(controller) == before
    _close_1d(controller)
    # Close cancels an adopted replacement reservation before any submit.
    controller = _controller(); active[0] = controller
    controller.open_viewer_1d((paths[0],)); _await_ready(controller)
    submits.clear(); mints.clear(); transport_calls.clear(); blocked.append(True)
    clear = controller.begin_viewer_1d_renderer_clear((paths[1],))
    assert not controller.acknowledge_viewer_1d_renderer_clear(
        viewer._new_viewer_1d_renderer_clear_receipt(clear, True))
    assert controller.open_viewer_1d((paths[2],)) is None
    gate = controller.viewer_1d_context.commit_gate
    assert not controller.close_viewer_1d()
    assert gate.cancelled and controller._viewer_1d.latest_intent is None
    _close_1d(controller)
    assert submits == [] and mints == []
    assert {name for name, _locked in transport_calls} >= {"cancel_gate", "retains_gate", "retire"}
    assert not any(locked for _name, locked in transport_calls)
    monkeypatch.undo()
    # The transport-held A cleanup path preserves the exact latest C request.
    from tests.xdart.scattering import test_p2b0_viewer_transport as b0
    controller = _controller()
    b0._force_real_cleanup_pending(viewer, monkeypatch)
    cleanup_calls = []
    for name in ("blocked_cleanup_token", "retry_blocked_cleanup"):
        operation = getattr(context_module.HydrationTransport, name)
        def probe_cleanup(raw, *args, _name=name, _operation=operation):
            cleanup_calls.append((_name, controller._viewer_2d_lock._is_owned()))
            return _operation(raw, *args)
        monkeypatch.setattr(context_module.HydrationTransport, name, probe_cleanup)
    a = controller.open_viewer_1d((paths[0],))
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and controller._viewer_1d.cleanup_notice is None:
        time.sleep(0.005)
    assert controller._viewer_1d.cleanup_notice.request is a
    assert cleanup_calls == []
    transitions, mint_epochs = [], []
    real_complete = type(controller._viewer_1d).complete
    real_request = ContextController._viewer_1d_request
    def complete(owner, completion):
        result = real_complete(owner, completion)
        if completion.token is a.token:
            transitions.append((owner.context.state.value, owner.request, owner.cleanup_notice))
        return result
    def mint(owner, context, policy, provider):
        mint_epochs.append((context.commit_gate._reserved_epoch, context.commit_gate.epoch))
        return real_request(owner, context, policy, provider)
    monkeypatch.setattr(type(controller._viewer_1d), "complete", complete)
    monkeypatch.setattr(ContextController, "_viewer_1d_request", mint)
    b = controller.open_viewer_1d((paths[1],))
    c = controller.open_viewer_1d((paths[2],))
    assert b is not c and controller._viewer_1d.request is c
    assert controller._viewer_1d.cleanup_notice.request is a
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.poll_viewer_1d()
        if controller.viewer_1d_context.state.value == "ready":
            break
        time.sleep(0.005)
    assert controller.viewer_1d_context.state.value == "ready"
    assert controller._viewer_1d.request is c
    assert transitions == [("loading", c, None)]
    assert mint_epochs == [(0, b.read_key.scope.epoch), (0, c.read_key.scope.epoch)]
    assert {name for name, _locked in cleanup_calls} == {"blocked_cleanup_token", "retry_blocked_cleanup"}
    assert not any(locked for _name, locked in cleanup_calls)
    before = _snapshot(controller)
    controller._viewer_1d.complete(HydrationCompletion(
        a.token, HydrationOutcome.FAILED, "duplicate A",
    ))
    assert _snapshot(controller) == before
    _close_1d(controller)
    # An unadopted A with no successor terminates EMPTY; cancelled stays CLOSED.
    monkeypatch.undo(); b0._force_real_cleanup_pending(viewer, monkeypatch)
    controller = _controller(); a = controller.open_viewer_1d((paths[0],))
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and controller._viewer_1d.cleanup_notice is None:
        time.sleep(0.005)
    notice = controller._viewer_1d.cleanup_notice
    assert notice.request is a
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and controller.viewer_1d_context.state.value != "empty":
        controller.poll_viewer_1d(); time.sleep(0.005)
    before = _snapshot(controller)
    controller._viewer_1d.complete(HydrationCompletion(a.token, HydrationOutcome.FAILED, "duplicate"))
    assert controller.viewer_1d_context.state.value == "empty" and _snapshot(controller) == before
    _close_1d(controller)
    controller._viewer_1d.cleanup_pending(notice)
    assert controller.viewer_1d_context is None
    monkeypatch.undo(); b0._force_real_cleanup_pending(viewer, monkeypatch)
    controller = _controller(); a = controller.open_viewer_1d((paths[0],))
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and controller._viewer_1d.cleanup_notice is None:
        time.sleep(0.005)
    provider = controller._viewer_1d.provider
    outer = provider.blocked_cleanup_token(a.token)
    real_retry = type(provider).retry_blocked_cleanup
    monkeypatch.setattr(type(provider), "retry_blocked_cleanup", lambda *_args: None)
    assert not controller.close_viewer_1d()
    controller._viewer_1d.complete(HydrationCompletion(a.token, HydrationOutcome.FAILED, "closed"))
    assert controller.viewer_1d_context.state.value == "closed" and controller.viewer_1d_cleanup_pending
    monkeypatch.setattr(type(provider), "retry_blocked_cleanup", real_retry)
    real_retry(provider, outer); _close_1d(controller)


def test_viewer_1d_modes_preserve_sigma_and_axes_and_refuse_invalid_combination(tmp_path, monkeypatch) -> None:
    import xdart.gui.tabs.scattering.context_projection as projection_module
    (tmp_path / "left").mkdir(); (tmp_path / "right").mkdir()
    first = _write_xye(tmp_path / "left/same.xye", [0, 1 + 1.5e-12, 2], [10, 11, 12], [.5, .6, .7])
    second = _write_xye(tmp_path / "right/same.xye", [.5, 1.5], [20, 21], [.8, .9])
    controller = _controller()
    _api(controller, "open_viewer_1d")((str(first), str(second)))
    _await_ready(controller)
    owner = controller._viewer_1d
    borrow = owner.holder.borrow
    request, gate, generation = owner.request, owner.context.commit_gate, owner.context.generation
    provider, provider_type, counters = owner.provider, type(owner.provider), owner.provider.counters()
    resident_effects, real_mint, real_open = [], ContextController._viewer_1d_request, projection_module.os.open
    provider_calls = {name: getattr(provider_type, name) for name in ("completions", "counters", "submit", "submit_detached", "dispatch_detached", "cancel_gate", "cancel_gate_detached", "retains_gate", "retire", "blocked_cleanup_token", "retry_blocked_cleanup")}
    for name, operation in provider_calls.items(): monkeypatch.setattr(provider_type, name, lambda *args, _name=name, _operation=operation, **kwargs: (resident_effects.append(("provider", _name)), _operation(*args, **kwargs))[1])
    monkeypatch.setattr(ContextController, "_viewer_1d_request", lambda *args, **kwargs: (resident_effects.append(("mint", None)), real_mint(*args, **kwargs))[1])
    monkeypatch.setattr(projection_module.os, "open", lambda *args, **kwargs: (resident_effects.append(("io", args[0] if args else None)), real_open(*args, **kwargs))[1])
    single = _shell(controller, ScientificPreferences(plot_mode="Single"))
    assert single.scientific.processing_mode == "1D Viewer"
    assert single.scientific.plot_mode == "Single"
    assert single.browser.frames == controller.navigation.frames
    assert single.browser.selected_scan == str(first)
    assert single.browser.selected_artifacts == (str(first), str(second))
    assert single.browser.multi_artifact_selection
    assert len(single.scientific.traces) == 1
    trace = single.scientific.traces[0]
    mode = borrow.modes[0]
    assert trace.axis.values is mode.coordinate
    assert trace.intensity is mode.intensity
    assert trace.sigma is mode.uncertainty
    assert not trace.sigma.flags.writeable
    frames = controller.navigation.frames
    assert (frames[0] is not frames[1] and owner.request.paths == (str(first), str(second)) and controller.navigation.current is frames[0] and controller.navigation.selected == frames)
    assert controller.select_viewer_1d(frames[1], (frames[1],))
    assert controller.navigation.current is frames[1]
    assert controller.navigation.selected == (frames[1],)
    selected = _shell(controller, ScientificPreferences(plot_mode="Single"))
    assert len(selected.scientific.traces) == 1
    assert selected.scientific.traces[0].frame is frames[1]
    assert selected.browser.selected_scan == str(second)
    assert selected.browser.selected_artifacts == (str(second),)
    selected_by_context = _shell(controller, ScientificPreferences(plot_mode="Single"), "Int 2D")
    assert selected_by_context.scientific.processing_mode == "1D Viewer"
    commands = []
    view = SimpleNamespace(
        _processing_mode="1D Viewer", _frame_keys=frames,
        _selected_keys=(frames[1],), _single_mode=True,
        frame_selector=SimpleNamespace(itemData=lambda _index: frames[0]),
        commandRequested=SimpleNamespace(emit=commands.append),
    )
    ScientificView._frame_selected(view, 0)
    assert commands[0].frames == (frames[0],)
    presented = []
    page = SimpleNamespace(
        _context_controller=controller,
        _refresh_shell=lambda: presented.append(
            _shell(controller, ScientificPreferences()).scientific.traces[0].frame),
    )
    ScatteringWorkspace._select_frames(page, commands[0])
    assert presented == [frames[0]] and controller.navigation.selected == (frames[0],)
    current_only = _shell(controller, ScientificPreferences(plot_mode="Single"))
    assert current_only.browser.selected_scan == str(first)
    assert current_only.browser.selected_artifacts == (str(first),)
    assert controller.select_viewer_1d(frames[1], frames) and controller.navigation.current is frames[1]
    overlay = _shell(controller, ScientificPreferences(plot_mode="Overlay"))
    assert tuple(trace.frame for trace in overlay.scientific.traces) == frames
    assert overlay.scientific.traces[0].axis.values is borrow.modes[0].coordinate
    assert overlay.scientific.traces[1].axis.values is borrow.modes[1].coordinate
    assert overlay.scientific.traces[0].axis.values.shape != overlay.scientific.traces[1].axis.values.shape
    ScatteringWorkspace._select_frames(page, ShellCommand(
        ShellCommandKind.SELECT_BROWSER_FRAMES,
        frame=frames[0],
        frames=(frames[0],),
        intent=FrameSelectionIntent.TOGGLE_TRACE,
    ))
    assert controller.navigation.current is frames[0]
    assert controller.navigation.selected == (frames[1],)
    toggled = _shell(
        controller, ScientificPreferences(plot_mode="Overlay")
    )
    assert tuple(
        item.frame for item in toggled.scientific.traces
    ) == (frames[1],)
    assert toggled.browser.selected_scan == str(first)
    assert toggled.browser.selected_artifacts == (str(second),)
    assert controller.select_viewer_1d(frames[1], frames)
    waterfall = _shell(controller, ScientificPreferences(plot_mode="Waterfall"))
    assert "first selected 1D source" in waterfall.scientific.status
    assert all(trace.axis.values is borrow.modes[0].coordinate for trace in waterfall.scientific.traces)
    assert np.isnan(waterfall.scientific.traces[1].intensity[0])
    assert np.isnan(waterfall.scientific.traces[1].intensity[-1])
    assert not waterfall.scientific.traces[1].intensity.flags.writeable
    assert np.isnan(waterfall.scientific.traces[1].sigma[[0, -1]]).all() and not waterfall.scientific.traces[1].sigma.flags.writeable
    assert np.array_equal(borrow.modes[0].coordinate, [0, 1 + 1.5e-12, 2])
    assert np.array_equal(borrow.modes[1].coordinate, [.5, 1.5])
    assert np.array_equal(borrow.modes[1].intensity, [20, 21]) and np.array_equal(borrow.modes[1].uncertainty, [.8, .9])
    rendered, plots = [], []
    legend = SimpleNamespace(setVisible=lambda _value: None)
    curve = SimpleNamespace(listDataItems=lambda: (), clear=lambda: None,
        getPlotItem=lambda: SimpleNamespace(legend=legend), addLegend=lambda: legend,
        plot=lambda *args, **_kwargs: plots.append(args), setLabel=lambda *_args, **_kwargs: None)
    render = SimpleNamespace(_processing_mode="1D Viewer", _trace_history_scope=(), _trace_selection_keys=(),
        _trace_history_by_identity={}, _trace_history_keys=(), _bottom_waterfall_active=False, _share_link_on=False,
        _rendered_plot_mode="", _rendered_plot_options=None, _rendered_overlay_step=None,
        _rendered_trace_keys=(), _rendered_trace_axis_key=None,
        _waterfall_source_keys=(), _waterfall_render_contract=None,
        _rendered_browse_science_contract=None,
        _merge_pinned_trace_history=lambda _state: (), _skip_live_waterfall=lambda *_args, **_kwargs: False,
        _bounded_waterfall_rows=lambda rows: rows, _axis_key=lambda axis: id(axis.values),
        _waterfall_axis=lambda _scope, traces, *_args: (np.arange(len(traces), dtype=float), "Frame #"),
        waterfall=SimpleNamespace(render=lambda rows, **kwargs: rendered.append((rows, kwargs))),
        curve=curve, legend=legend, bottom_stack=SimpleNamespace(
            setCurrentWidget=lambda _widget: None,
            currentWidget=lambda: curve,
        ))
    render._merge_trace_history = partial(ScientificView._merge_trace_history, render); render._render_waterfall = partial(ScientificView._render_waterfall, render)
    assert render._merge_trace_history(single.scientific, controller.navigation)[0] is trace
    assert render._merge_trace_history(selected.scientific, controller.navigation)[0] is selected.scientific.traces[0] and render._trace_history_keys == (frames[1],)
    monkeypatch.setattr("xdart.gui.tabs.scattering.scientific_view.resample_image_axis_to_uniform",
                        lambda *_args, **_kwargs: pytest.fail("viewer grid was resampled"))
    waterfall_state = replace(waterfall.scientific, plot_options=replace(waterfall.scientific.plot_options, waterfall_start=2))
    ScientificView._render_traces(render, waterfall_state, controller.navigation, live_update=False)
    assert rendered[0][0].shape == (2, 3)
    assert rendered[0][1]["x_axis"].values is borrow.modes[0].coordinate
    real_merge = render._merge_trace_history; render._merge_trace_history = lambda state, _nav: state.traces; overlay_state = replace(overlay.scientific, traces=(overlay.scientific.traces[0],) * 16,
        plot_options=replace(overlay.scientific.plot_options, waterfall_start=4, waterfall_stop=6, waterfall_step=2))
    ScientificView._render_traces(render, overlay_state, controller.navigation, live_update=False)
    assert len(plots) == 16 and not render._bottom_waterfall_active; render._merge_trace_history = real_merge
    monkeypatch.setattr("xdart.gui.tabs.scattering.scientific_view.aggregate_traces",
                        lambda *_args: pytest.fail("viewer entered native aggregate"))
    for aggregate in ("Average", "Sum"):
        preferences = ScientificPreferences(plot_mode=aggregate)
        inherited = _shell(controller, preferences)
        assert (inherited.scientific.plot_mode == "Single" and preferences.plot_mode == aggregate
                and inherited.scientific.slice_pins == () and inherited.scientific.pinned_traces == ())
        assert len(inherited.scientific.traces) == 1
        ScientificView._render_traces(render, inherited.scientific, controller.navigation, live_update=False)
    render._merge_trace_history = render._render_waterfall = None; rendered.clear(); plots.clear()
    payloads = controller.project_navigation(preferences=ScientificPreferences(), processing_mode="1D Viewer")
    cases = (([0, 1 + 4e-12, 2], 0, "nonuniform"), ([0, np.nan, 2], 0, "finite"),
             ([2, 1, 0], 0, "strictly increasing"), ([0, 2, 1], 0, "strictly increasing"),
             ([0, 2, 1], 1, "strictly increasing"), ([0], 0, "finite"),
             ([0, 1, 1], 0, "strictly increasing"))
    for axis, index, diagnostic in cases:
        base = payloads[index]; values = np.asarray(axis, dtype=float)
        view = replace(base.view, axis_1d=replace(base.view.axis_1d, values=values),
                       intensity_1d=np.ones(len(values)), sigma_1d=None)
        invalid = list(payloads); invalid[index] = replace(base, view=view)
        refused = projection_module._viewer_1d_scientific(controller.navigation, tuple(invalid), controller.resident_frame_keys, ScientificPreferences(plot_mode="Waterfall"), "")
        assert refused.traces == () and diagnostic in refused.status
    assert owner.request is request and owner.context.commit_gate is gate and gate.epoch == request.read_key.scope.epoch and gate._reserved_epoch == 0 and owner.context.generation == generation and owner.provider is provider
    assert provider_calls["counters"](provider) == counters and resident_effects == []
    monkeypatch.undo()
    clear = controller.begin_viewer_1d_renderer_clear()
    render = SimpleNamespace(
        _trace_history_by_identity={id(trace): trace}, _pinned_trace_by_id={id(trace): trace},
        _trace_history_keys=(trace.frame,), _rendered_trace_keys=(("live", id(trace.frame)),), _waterfall_y_values=(1.0,),
        curve=SimpleNamespace(listDataItems=lambda: ()),
        title=SimpleNamespace(setText=lambda _value: None),
        status=SimpleNamespace(setText=lambda _value: None),
    )
    def scrub(target, _request, *, failure):
        assert failure and target._trace_history_by_identity[id(trace)].sigma is mode.uncertainty
        target._trace_history_by_identity.clear(); target._pinned_trace_by_id.clear(); target._trace_history_keys = ()
        target._rendered_trace_keys = (); target._waterfall_y_values = (); return True
    monkeypatch.setattr(ScientificView, "clear_viewer_2d", scrub)
    receipt = ScientificView.clear_viewer_1d(render, clear)
    assert receipt.cleared and not render._trace_history_by_identity and not render._pinned_trace_by_id and render._waterfall_y_values == ()
    del single, trace, mode, selected, selected_by_context, current_only, overlay, toggled, waterfall, inherited, borrow, waterfall_state, overlay_state, render, real_merge, payloads, base, view, invalid, refused
    assert controller.acknowledge_viewer_1d_renderer_clear(receipt)
    _close_1d(controller)

    mixed = tmp_path / "mixed.npz"
    np.savez(mixed, x=np.arange(3.0), y=np.arange(3.0),
             x_unit=np.asarray("degrees"))
    controller = _controller()
    controller.open_viewer_1d((str(first), str(mixed)))
    _await_ready(controller)
    frames = controller.navigation.frames
    assert controller.select_viewer_1d(frames[0], frames)
    refused = _shell(controller, ScientificPreferences(plot_mode="Overlay"))
    assert refused.scientific.traces == ()
    assert "conflicting units" in refused.scientific.status
    assert controller.viewer_1d_context.state.value == "ready"
    _close_1d(controller)


def test_viewer_1d_cross_context_switch_and_workspace_close_are_positive(tmp_path) -> None:
    import xrd_tools.session.viewer_1d as viewer
    from tests.xdart.scattering.test_e3_context_contract import _cold_controller, _select_browse
    one_d = _write_xye(tmp_path / "one.xye", [0, 1], [1, 2])
    two_d = tmp_path / "two.npy"
    np.save(two_d, np.arange(6.0).reshape(2, 3))
    controller, _lifecycle, loader = _cold_controller()
    _api(controller, "open_viewer_1d")((str(one_d),))
    _await_ready(controller)
    with pytest.raises(RuntimeError, match="cannot adopt acquisition"): controller.adopt_acquisition(RunIdentity(9, "blocked"))
    with pytest.raises(RuntimeError, match="1D Viewer renderer clear"):
        controller.open_viewer_2d(str(two_d))
    with pytest.raises(RuntimeError, match="1D Viewer cleanup"):
        controller.begin_browse(str(one_d))
    assert _positive_clear(controller)
    assert controller.close_viewer_1d()
    _request, browse = _select_browse(controller, loader)
    assert controller.selection.kind is ContextKind.BROWSE
    controller.open_viewer_1d((str(one_d),))
    assert browse in loader.released
    _await_ready(controller); assert _positive_clear(controller)
    assert controller.close_viewer_1d()
    controller.open_viewer_2d(str(two_d))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.poll_viewer_2d()
        if controller.viewer_2d_frame is not None:
            break
        time.sleep(0.005)
    assert controller.viewer_2d_frame is not None
    with pytest.raises(RuntimeError, match="2D Viewer renderer clear"):
        controller.open_viewer_1d((str(one_d),))
    clear = controller.begin_viewer_2d_renderer_clear()
    from xdart.modules.display_context import Viewer2DRendererClearReceipt
    assert controller.acknowledge_viewer_2d_renderer_clear(
        Viewer2DRendererClearReceipt(clear, True)
    )
    assert controller.close_viewer_2d()
    controller.open_viewer_1d((str(one_d),))
    _await_ready(controller)
    clear_results = [True, False, True]
    workspace = SimpleNamespace(_context_controller=controller,
        _shell=SimpleNamespace(browser=SimpleNamespace(reconcile_heavy_residency=lambda *_args, **_kwargs: None), scientific=SimpleNamespace(clear_viewer_1d=lambda request:
            viewer._new_viewer_1d_renderer_clear_receipt(request, clear_results.pop(0)))),
        _last_scientific_projection=object(), _preferences=ScientificPreferences(),
        _terminal_close=None, _closing=True, _lifecycle=SimpleNamespace(phase=RunPhase.IDLE),
        _close_identity=None, _clear_viewer_2d_renderer=lambda *, close: True)
    workspace._clear_viewer_1d_renderer = partial(ScatteringWorkspace._clear_viewer_1d_renderer, workspace)
    workspace.__dict__.update(_closed=False, _sync_detector_demand=lambda: None, _retain_outgoing_display=False, _source_observation=None,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(thaw=lambda: SimpleNamespace(
            processing_mode="1D Viewer", live_mode=False, source_spec=None, run_options={}))), _project_controls=lambda _snapshot: None,
        _start_permitted=lambda: (True, ""), _context_projection=SimpleNamespace(build_shell=lambda **_: object()),
        _shell_revision=0, _date_sorted=False, _auto_last=False, _run_executor=None, _notice_text="",
        **dict.fromkeys(("_controls_readiness", "_progress", "_browser_directory", "_browser_catalog", "_browser_transient_frame")))
    workspace._lifecycle.__dict__.update(reset_permitted=False, active_run_identity=None, attempt_run_identity=None)
    notices = []; workspace._notice = notices.append; workspace._shell.apply_state = lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError("render"))
    ScatteringWorkspace._refresh_shell(workspace)
    assert controller.viewer_1d_cleanup_pending and workspace._last_scientific_projection is None and "Passive shell render failed" in notices[-1]
    assert workspace._clear_viewer_1d_renderer(close=True) and controller.viewer_1d_context is None; controller.open_viewer_1d((str(one_d),)); _await_ready(controller)
    pending = ScatteringWorkspace.close_workspace(workspace)
    assert pending.cleanup_status.value == "cleanup_pending" and controller.viewer_1d_cleanup_pending
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not ScatteringWorkspace._edit_scientific_preference(
            workspace, ShellCommand(ShellCommandKind.CLEAR_1D)):
        time.sleep(0.005)
    assert controller.viewer_1d_context is None


def test_catalog_activation_routes_are_gui_thread_zero_io(monkeypatch) -> None:
    calls = []
    intent = SimpleNamespace(processing_mode="1D Viewer")
    controller = SimpleNamespace(
        viewer_1d_owned=False, viewer_2d_owned=False, selection=None,
    )
    page = SimpleNamespace(
        _closing=False,
        _closed=False,
        _context_controller=controller,
        _operation_slot=SimpleNamespace(
            owned=False, current_identity=None, observe_stamp=lambda _stamp: None,
        ),
        _calibration_identity=None,
        _mask_identity=None,
        _reintegrate_identity=None,
        _reintegrate_dimension=None,
        _experiment_operation_busy=lambda: False,
        _notice=lambda _message: None,
        _refresh_shell=lambda: None,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(
            revision=1, thaw=lambda: intent,
        )),
        _open_viewer_1d_paths=lambda paths, *, current_path=None: calls.append(
            ("xye", paths, current_path)
        ),
        _open_viewer_2d_path=lambda path: calls.append(("tiff", path)),
        _clear_viewer_1d_renderer=lambda **_kwargs: True,
        _clear_viewer_2d_renderer=lambda **_kwargs: True,
        _select_scan=lambda value, **kwargs: calls.append(
            ("select", value, kwargs)
        ),
    )
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("SELECT_SCAN performed GUI-thread filesystem I/O")
    )
    monkeypatch.setattr(page_module.os.path, "isdir", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)

    ScatteringWorkspace._handle_shell_command(page, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "/data/subdir", path=("directory",),
    ))
    ScatteringWorkspace._handle_shell_command(page, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "/data/second.xye", path=("artifact",),
        artifacts=("/data/curve.xye", "/data/second.xye"),
    ))
    intent.processing_mode = "2D Viewer"
    ScatteringWorkspace._handle_shell_command(page, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "/data/image.tiff", path=("artifact",),
    ))
    intent.processing_mode = "Int 2D"
    ScatteringWorkspace._handle_shell_command(page, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "/data/result.nxs", path=("artifact",),
    ))

    assert calls == [
        ("select", "/data/subdir", {"is_directory": True}),
        (
            "xye",
            ("/data/curve.xye", "/data/second.xye"),
            "/data/second.xye",
        ),
        ("tiff", "/data/image.tiff"),
        ("select", "/data/result.nxs", {"is_directory": False}),
    ]


def test_viewer_1d_catalog_selection_emits_one_exact_batch_command() -> None:
    from xdart.gui.tabs.scattering.browser_view import (
        BrowserView,
        _DIRECTORY_ROLE,
        _USER_ROLE,
    )

    class _Artifact:
        def __init__(self, path: str, *, selected: bool = True) -> None:
            self.path = path
            self.selected = selected

        def data(self, role: int):
            if role == _USER_ROLE:
                return self.path
            if role == _DIRECTORY_ROLE:
                return False
            raise AssertionError("unexpected browser role")

        def isSelected(self):
            return self.selected

    commands: list[ShellCommand] = []
    paths = ("/data/first.xye", "/data/second.csv")
    items = tuple(_Artifact(path) for path in paths)
    view = SimpleNamespace(
        scans=SimpleNamespace(
            count=lambda: len(items),
            item=lambda index: items[index],
            currentItem=lambda: items[0],
        ),
        _cancel_pending_frame_selection=lambda: None,
        commandRequested=SimpleNamespace(emit=commands.append),
    )

    BrowserView._scan_selected(view)

    assert commands == [ShellCommand(
        ShellCommandKind.SELECT_SCAN,
        paths[0],
        path=("artifact",),
        artifacts=paths,
    )]


def test_viewer_1d_clicked_current_seeds_single_but_overlay_keeps_all_paths(
    tmp_path,
) -> None:
    from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

    from xdart.gui.tabs.scattering.browser_view import BrowserView
    from xdart.gui.tabs.scattering.shell_projection import (
        build_browser_projection,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = _write_xye(tmp_path / "a.xye", [0, 1], [10, 11])
    second = _write_xye(tmp_path / "b.xye", [0, 1], [20, 21])
    paths = (str(first), str(second))
    catalog = tuple(
        BrowserCatalogEntry(path, Path(path).name, index)
        for index, path in enumerate(paths)
    )
    controller = _controller()
    intent = SimpleNamespace(processing_mode="1D Viewer")
    page = SimpleNamespace(
        _closing=False,
        _closed=False,
        _context_controller=controller,
        _operation_slot=SimpleNamespace(
            owned=False,
            current_identity=None,
            observe_stamp=lambda _stamp: None,
        ),
        _experiment_operation_busy=lambda: False,
        _calibration_identity=None,
        _mask_identity=None,
        _reintegrate_identity=None,
        _reintegrate_dimension=None,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(
            revision=1,
            thaw=lambda: intent,
        )),
        _retire_batch_presentation=lambda: None,
        _retain_outgoing_display=True,
        _clear_viewer_2d_renderer=lambda **_kwargs: True,
        _notice=lambda _message: None,
        _refresh_shell=lambda: None,
        _ensure_timer=lambda: None,
        _error_notice=lambda title, error: pytest.fail(f"{title}: {error}"),
    )
    page._open_viewer_1d_paths = partial(
        ScatteringWorkspace._open_viewer_1d_paths,
        page,
    )
    browser = BrowserView()
    browser.resize(640, 480)
    browser.show()
    browser.commandRequested.connect(partial(
        ScatteringWorkspace._handle_shell_command,
        page,
    ))
    browser.reconcile(
        build_browser_projection(
            contexts=(),
            selection=None,
            navigation=FrameNavigationProjection(),
            browser_directory=str(tmp_path),
            date_sorted=False,
            auto_last=True,
            catalog=catalog,
            selected_artifacts=(paths[0],),
            current_artifact=paths[0],
            multi_artifact_selection=True,
        ),
        FrameNavigationProjection(),
        plot_mode="Single",
    )
    app.processEvents()
    second_row = browser.scans.visualItemRect(browser.scans.item(1)).center()
    QtTest.QTest.mouseClick(
        browser.scans.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        second_row,
    )
    app.processEvents()
    _await_ready(controller)
    single = overlay = None
    try:
        frames = controller.navigation.frames
        assert controller.viewer_1d_context.paths == paths
        assert controller.viewer_1d_context.current_path == paths[1]
        assert controller.navigation.current is frames[1]
        assert controller.navigation.selected == frames

        single = _shell(
            controller,
            ScientificPreferences(plot_mode="Single"),
            browser_catalog=catalog,
        )
        assert single.browser.selected_scan == paths[1]
        assert single.browser.selected_artifacts == paths
        assert tuple(trace.frame for trace in single.scientific.traces) == (
            frames[1],
        )

        overlay = _shell(
            controller,
            ScientificPreferences(plot_mode="Overlay"),
        )
        assert tuple(trace.frame for trace in overlay.scientific.traces) == frames
        assert tuple(trace.title for trace in overlay.scientific.traces) == (
            first.name,
            second.name,
        )
    finally:
        del single, overlay
        _close_1d(controller)
        browser.deleteLater()
        app.processEvents()


def test_viewer_1d_single_collapses_membership_without_preclear(
    monkeypatch,
) -> None:
    current = object()
    calls: list[object] = []
    controller = SimpleNamespace(
        selection=SimpleNamespace(kind=ContextKind.VIEWER_1D),
        navigation=SimpleNamespace(current=current),
        select_viewer_1d=lambda frame, selected: calls.append(
            ("select", frame, selected)
        ) or True,
    )
    page = SimpleNamespace(
        _context_controller=controller,
        _preferences=ScientificPreferences(plot_mode="Waterfall"),
        _retire_batch_presentation=lambda: calls.append("retire"),
        _shell=SimpleNamespace(browser=SimpleNamespace(
            cancel_pending_frame_selection=lambda: calls.append("cancel"),
        )),
        _background_owner=SimpleNamespace(projection=lambda: None),
    )
    monkeypatch.setattr(
        ScatteringWorkspace,
        "_clear_presentation_targets",
        lambda *_args: pytest.fail("Viewer1D mode switch pre-cleared plots"),
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        page,
        ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Single"),
    )

    assert calls == ["retire", "cancel", ("select", current, (current,))]
    assert page._preferences.plot_mode == "Single"


def test_viewer_1d_page_commands_reload_status_and_native_restore(monkeypatch, tmp_path) -> None:
    calls = []
    paths = ("/opaque/first.xye", "/other/first.xye")
    context = SimpleNamespace(
        paths=paths,
        current_path=paths[1],
        state=SimpleNamespace(value="ready"),
    )
    controller = SimpleNamespace(viewer_1d_context=None, viewer_1d_owned=False,
        viewer_2d_owned=False, run_identity=None,
        open_viewer_1d=lambda selected, *, current_path=None: calls.append(
            ("open", selected, current_path)
        ) or object())
    intent = SimpleNamespace(processing_mode="1D Viewer", live_mode=False, run_options={})
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("foreign seam"))
    page = SimpleNamespace(_context_controller=controller,
        _lifecycle=SimpleNamespace(phase=RunPhase.IDLE),
        _retire_batch_presentation=lambda: None,
        _retain_outgoing_display=True,
        _viewer_1d_start_directory=lambda: calls.append("start") or "/viewer",
        _viewer_file_chooser=lambda start: calls.append(("choose", start)) or paths,
        _clear_viewer_1d_renderer=forbidden, _intents=SimpleNamespace(
            snapshot=lambda: SimpleNamespace(thaw=lambda: intent)),
        _notice=lambda value: calls.append(("notice", value)), _ensure_timer=lambda: calls.append("timer"),
        _error_notice=forbidden, _begin_run=forbidden, _commit_focused_control_edit_for_run=forbidden)
    page._choose_viewer_1d_files = partial(
        _api(ScatteringWorkspace, "_choose_viewer_1d_files"), page)
    page._open_viewer_1d_paths = partial(
        _api(ScatteringWorkspace, "_open_viewer_1d_paths"), page)
    ScatteringWorkspace._run_action(page)
    assert calls == [
        "start",
        ("choose", "/viewer"),
        ("open", paths, paths[0]),
        ("notice", ""),
        "timer",
    ]
    controller.viewer_1d_context = context
    controller.viewer_1d_owned = True
    page._viewer_file_chooser = forbidden
    page._clear_viewer_1d_renderer = lambda *, paths=None, current_path=None, close=False: (
        calls.append(("clear", paths, current_path, close)) or True
    )
    ScatteringWorkspace._run_action(page)
    assert calls[-2:] == [("clear", paths, paths[1], False), "timer"]
    page._clear_viewer_1d_renderer = lambda **_: False
    ScatteringWorkspace._edit_run_strip(page, ShellCommandKind.SET_PROCESSING_MODE, "Int 2D")
    assert intent.processing_mode == "1D Viewer" and calls[-1] == ("notice", "1D Viewer cleanup remains pending")
    ordered, outcomes = [], [False, True]
    directory = tmp_path / "viewer-folder"; directory.mkdir()
    dispatch = SimpleNamespace(_closing=False, _closed=False,
        _context_controller=SimpleNamespace(viewer_1d_owned=True, viewer_2d_owned=False, selection=None),
        _operation_slot=SimpleNamespace(owned=False, current_identity=None, observe_stamp=lambda _stamp: None),
        _experiment_operation_busy=lambda: False,
        _notice=lambda _message: None,
        _refresh_shell=lambda: None,
        _calibration_identity=None, _mask_identity=None,
        _reintegrate_identity=None, _reintegrate_dimension=None,
        _clear_viewer_1d_renderer=lambda *, close: ordered.append(("clear", close)) or outcomes.pop(0),
        _clear_viewer_2d_renderer=forbidden,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(revision=1, thaw=lambda: intent)),
        _open_viewer_1d_paths=lambda value, *, current_path=None: ordered.append(
            ("open-1d", value, current_path)
        ),
        _open_viewer_2d_path=lambda value: ordered.append(("open-2d", value)),
        _select_scan=lambda value, **kwargs: ordered.append(
            ("select", value, kwargs)
        ))
    ScatteringWorkspace._handle_shell_command(
        dispatch, ShellCommand(
            ShellCommandKind.SELECT_SCAN, str(directory), path=("directory",),
        ))
    assert ordered == [("select", str(directory), {"is_directory": True})]
    ScatteringWorkspace._handle_shell_command(dispatch, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "scan.xye", path=("artifact",),
    ))
    assert ordered[-1] == ("open-1d", ("scan.xye",), "scan.xye")
    intent.processing_mode = "2D Viewer"
    ScatteringWorkspace._handle_shell_command(dispatch, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "image.tif", path=("artifact",),
    ))
    assert ordered[-1] == ("open-2d", "image.tif")
    intent.processing_mode = "Int 2D"
    ScatteringWorkspace._handle_shell_command(dispatch, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "scan.nxs", path=("artifact",),
    ))
    assert ordered[-1] == ("clear", True)
    ScatteringWorkspace._handle_shell_command(dispatch, ShellCommand(
        ShellCommandKind.SELECT_SCAN, "scan.nxs", path=("artifact",),
    ))
    assert ordered[-2:] == [
        ("clear", True), ("select", "scan.nxs", {"is_directory": False}),
    ]
    identity = RunIdentity(7, "owned")
    dispatch._context_controller.__dict__.update(run_identity=identity, poll_viewer_1d=lambda: False,
        poll_viewer_2d=lambda: False, poll_browse_preview=lambda: False, browse_pending=False, adopt_acquisition=forbidden)
    dispatch.__dict__.update(_poll_admission=lambda: False, _run_executor=SimpleNamespace(drain_events=lambda: (
        StandardRunEvent(identity, StandardEventKind.CONTEXT_READY),)), _lifecycle=SimpleNamespace(
        active_run_identity=identity, attempt_run_identity=None), _refresh_shell=forbidden,
        _pending_reintegrate_reload=None,
        _settle_browse_1d_before_drain=lambda: True,
        _terminal_browse_handoff=None,
        _terminal_browse_presentation=None,
        _terminal_browse_perf=None,
        _dispatch_deferred_metadata=lambda: page_module._OperationRefresh.NONE,
        _batch_ready_to_paint=lambda: None,
        _show_queued_authored_asset_confirmation=lambda: None,
        _batch_terminal=SimpleNamespace(active=False),
        _polling_needed=lambda: True, _run_timer=SimpleNamespace(stop=forbidden),
        _scientific_repaint_pending=False,
        _retain_outgoing_display=False, _waterfall_candidate_count=0,
        _shell=SimpleNamespace(scientific=SimpleNamespace(
            trace_row_count=0, bottom_waterfall_active=False)))
    monkeypatch.setattr(
        ScatteringWorkspace,
        "_advance_presentation_target",
        lambda _self: False,
    )
    dispatch._clear_presentation_targets = lambda: None; ScatteringWorkspace._drain_executor(dispatch)
    preferences = ScientificPreferences(plot_mode="Overlay")
    owner = SimpleNamespace(
        _preferences=preferences,
        _retire_batch_presentation=lambda: None,
        _retain_outgoing_display=True,
        _context_controller=SimpleNamespace(
            selection=SimpleNamespace(kind=ContextKind.VIEWER_1D),
            navigation=SimpleNamespace(current=object()),
        ),
        _shell=SimpleNamespace(browser=SimpleNamespace(
            cancel_pending_frame_selection=lambda: calls.append("cancel"))),
        _background_owner=SimpleNamespace(projection=lambda: None),
    )
    for command in (ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Average"),
                    ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Sum"),
                    ShellCommand(ShellCommandKind.PIN_SLICE)):
        assert not ScatteringWorkspace._edit_scientific_preference(owner, command)
        assert owner._preferences is preferences
    clear_calls = []
    owner._clear_viewer_1d_renderer = lambda *, close=False: clear_calls.append(close) or True
    assert ScatteringWorkspace._edit_scientific_preference(
        owner, ShellCommand(ShellCommandKind.CLEAR_1D)
    )
    assert clear_calls == [True]
    enabled = []
    item = SimpleNamespace(setEnabled=enabled.append)
    view = SimpleNamespace(_processing_mode="Int 2D", detector_controls=SimpleNamespace(setVisible=lambda _value: None), raw_popup_button=SimpleNamespace(setVisible=lambda _value: None), raw_popup_dialog=None,
        plot_mode=SimpleNamespace(findText=lambda value: value,
                                  model=lambda: SimpleNamespace(item=lambda _index: item)),
        image_splitter=SimpleNamespace(setVisible=lambda value: calls.append(("images", value))),
        raw=SimpleNamespace(setVisible=lambda value: calls.append(("raw", value))), cake=SimpleNamespace(
            setVisible=lambda value: calls.append(("cake", value))),
        vertical_splitter=SimpleNamespace(
            widget=lambda _index: SimpleNamespace(setVisible=lambda value: calls.append(("bottom", value))),
            setSizes=lambda value: calls.append(("sizes", value)),
        ),
        norm=SimpleNamespace(setVisible=lambda value: calls.append(("norm", value))), background=SimpleNamespace(
            setVisible=lambda value: calls.append(("background", value))),
        image_axis=SimpleNamespace(setVisible=lambda value: calls.append(("image-axis", value))),
        share_axis=SimpleNamespace(setVisible=lambda value: calls.append(("share", value))),
        slice=SimpleNamespace(setVisible=lambda value: calls.append(("slice", value))), slice_center=SimpleNamespace(
            setVisible=lambda value: calls.append(("center", value))), slice_width=SimpleNamespace(
            setVisible=lambda value: calls.append(("width", value))), pin=SimpleNamespace(
            setVisible=lambda value: calls.append(("pin", value))), _set_share_link=lambda value: calls.append(("link", value)))
    ScientificView._apply_processing_layout(view, "1D Viewer")
    assert view._processing_mode == "1D Viewer"
    assert ("images", False) in calls and ("bottom", True) in calls
    assert enabled == [False, False]
    ScientificView._apply_processing_layout(view, "Int 2D")
    assert view._processing_mode == "Int 2D"
    assert ("images", True) in calls and ("sizes", [500, 500]) in calls
    assert enabled == [False, False, True, True] and owner._preferences is preferences


def test_viewer_1d_scope_and_authority_census_stays_bounded() -> None:
    trees = {name: ast.parse((_ROOT / "src/xdart/gui/tabs/scattering" / name).read_text())
             for name in _PRODUCTION}
    nodes = [node for tree in trees.values() for node in ast.walk(tree)]
    owners = [node for node in nodes
              if isinstance(node, ast.ClassDef) and node.name == "_OneDViewerOwner"]
    assert len(owners) == 1
    identifiers = [(node.id if isinstance(node, ast.Name) else node.attr) for node in nodes
                   if isinstance(node, (ast.Name, ast.Attribute))]
    expected = {"RLock": 1, "Lock": 0, "HydrationTransport": 2, "_OneDViewerOwner": 1,
                "Thread": 0, "ThreadPoolExecutor": 4, "Queue": 0, "deque": 2}
    assert {name: identifiers.count(name) for name in expected} == expected
    calls = [(node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
             for node in nodes if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))]
    assert (calls.count("_new_viewer_1d_renderer_clear_request"), calls.count("Viewer1DRendererClearRequest"),
            calls.count("_new_viewer_1d_renderer_clear_receipt"), calls.count("Viewer1DRendererClearReceipt")) == (1, 0, 1, 0)
    for relative in ("io/viewer_1d.py", "session/viewer_1d.py"):
        source = (_ROOT / "src/xrd_tools" / relative).read_text()
        assert "from xdart" not in source and "import xdart" not in source
