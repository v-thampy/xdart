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
from xdart.gui.tabs.scattering.batch_terminal_presentation import (
    BatchTerminalPresentationController,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import RunIdentity, StandardEventKind, StandardRunEvent
from xdart.gui.tabs.scattering.metadata_operations import MetadataOperationOwner
from xdart.gui.tabs.scattering.external_tools import unavailable_external_tools
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
import xdart.gui.tabs.scattering.page as page_module
from xdart.gui.tabs.scattering.processed_browser import ProcessedBrowserOwner
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    FrameSelectionIntent,
    ProgressProjection,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_operations import (
    WorkspaceOperationOwner,
    WorkspaceRefreshEffect,
)
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
        viewer_1d_current_path=controller.viewer_1d_artifact_selection[0],
        viewer_1d_selected_paths=controller.viewer_1d_artifact_selection[1],
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


@pytest.fixture
def real_scientific_view():
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    yield view
    assert view.clear_workspace()
    view.close()
    view.deleteLater()
    app.processEvents()


def _reconcile_real_viewer(view, scientific, navigation):
    view.reconcile(scientific, navigation, completed=0,
                   total=len(navigation.frames), detail="Ready")


def _release_real_viewer(controller, view):
    request = controller.begin_viewer_1d_renderer_clear()
    assert request is not None
    receipt = view.clear_viewer_1d(request)
    assert receipt.cleared
    assert controller.acknowledge_viewer_1d_renderer_clear(receipt)
    assert controller.close_viewer_1d()


def test_viewer_1d_modes_preserve_sigma_and_axes_and_refuse_invalid_combination(tmp_path, monkeypatch, real_scientific_view) -> None:
    import xdart.gui.tabs.scattering.context_projection as projection_module
    render = real_scientific_view
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
    assert len(single.scientific.traces) == 2
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
    assert controller.select_viewer_1d(frames[0], (frames[0],))
    current_only = _shell(controller, ScientificPreferences(plot_mode="Single"))
    assert current_only.browser.selected_scan == str(first)
    assert current_only.browser.selected_artifacts == (str(first),)
    assert controller.select_viewer_1d(frames[1], frames) and controller.navigation.current is frames[1]
    overlay = _shell(controller, ScientificPreferences(plot_mode="Overlay"))
    assert tuple(trace.frame for trace in overlay.scientific.traces) == frames
    assert overlay.scientific.traces[0].axis.values is borrow.modes[0].coordinate
    assert overlay.scientific.traces[1].axis.values is borrow.modes[1].coordinate
    assert overlay.scientific.traces[0].axis.values.shape != overlay.scientific.traces[1].axis.values.shape
    assert controller.select_viewer_1d(frames[0], (frames[1],))
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
    for projected, source in zip(waterfall.scientific.traces, borrow.modes.values(), strict=True):
        assert projected.axis.values is source.coordinate
        assert projected.intensity is source.intensity
        assert projected.sigma is source.uncertainty
    del projected, source
    assert np.array_equal(borrow.modes[0].coordinate, [0, 1 + 1.5e-12, 2])
    assert np.array_equal(borrow.modes[1].coordinate, [.5, 1.5])
    assert np.array_equal(borrow.modes[1].intensity, [20, 21]) and np.array_equal(borrow.modes[1].uncertainty, [.8, .9])
    _reconcile_real_viewer(render, single.scientific,
                           FrameNavigationProjection(frames, frames[0], frames))
    # Viewer rows borrow provider arrays; the detached native-history accessor
    # intentionally excludes this row type. Inspect the mounted viewer history.
    assert render._trace_history_by_identity[id(frames[0])] is trace
    assert render._trace_history_by_identity[id(frames[0])].sigma is mode.uncertainty
    _reconcile_real_viewer(render, selected.scientific,
                           FrameNavigationProjection(frames, frames[1], (frames[1],)))
    assert render.trace_history_keys == (frames[1],)
    curve, = render.curve.listDataItems()
    np.testing.assert_array_equal(curve.getData()[0], borrow.modes[1].coordinate)
    np.testing.assert_array_equal(curve.getData()[1], borrow.modes[1].intensity)

    rendered = []
    real_waterfall_render = render.waterfall.render
    def observe_waterfall(rows, **kwargs):
        rendered.append((rows, kwargs))
        return real_waterfall_render(rows, **kwargs)
    monkeypatch.setattr(render.waterfall, "render", observe_waterfall)
    waterfall_state = replace(waterfall.scientific,
        plot_options=replace(waterfall.scientific.plot_options, waterfall_start=2))
    _reconcile_real_viewer(render, waterfall_state, controller.navigation)
    assert render.bottom_waterfall_active
    assert rendered[0][0].shape == (2, 3)
    # ScientificImagePane converts row-major science to pyqtgraph's columns.
    assert render.waterfall.image.image.shape == (3, 2)
    np.testing.assert_array_equal(render.waterfall.canvas.raw_image,
        np.stack((borrow.modes[0].intensity, np.interp(borrow.modes[0].coordinate,
            borrow.modes[1].coordinate, borrow.modes[1].intensity,
            left=np.nan, right=np.nan))).T)
    assert rendered[0][1]["x_axis"].values is borrow.modes[0].coordinate
    assert all(render._trace_history_by_identity[id(item.frame)].sigma is item.sigma
               for item in waterfall.scientific.traces)
    _reconcile_real_viewer(render, overlay.scientific, controller.navigation)
    assert not render.bottom_waterfall_active
    assert len(render.curve.listDataItems()) == 2
    for index, item in enumerate(render.curve.listDataItems()):
        np.testing.assert_array_equal(item.getData()[0], borrow.modes[index].coordinate)

    monkeypatch.setattr("xdart.gui.tabs.scattering.scientific_view.aggregate_traces",
                        lambda *_args: pytest.fail("viewer entered native aggregate"))
    for aggregate in ("Average", "Sum"):
        preferences = ScientificPreferences(plot_mode=aggregate)
        inherited = _shell(controller, preferences)
        assert (inherited.scientific.plot_mode == "Single" and preferences.plot_mode == aggregate
                and inherited.scientific.slice_pins == () and inherited.scientific.pinned_traces == ())
        assert len(inherited.scientific.traces) == 2
        _reconcile_real_viewer(render, inherited.scientific, controller.navigation)
        assert len(render.curve.listDataItems()) == 2
    payloads = controller.project_navigation(preferences=ScientificPreferences(), processing_mode="1D Viewer")
    cases = (([0, np.nan, 2], 0, "finite"),
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
        _reconcile_real_viewer(render, refused, controller.navigation)
        assert not render.curve.listDataItems()
        assert render.bottom_stack.currentWidget() is render.curve
        assert diagnostic in render.status.text()
    assert owner.request is request and owner.context.commit_gate is gate and gate.epoch == request.read_key.scope.epoch and gate._reserved_epoch == 0 and owner.context.generation == generation and owner.provider is provider
    assert provider_calls["counters"](provider) == counters and resident_effects == []
    monkeypatch.undo()
    _reconcile_real_viewer(render, waterfall.scientific, controller.navigation)
    receipt = render.clear_viewer_1d(controller.begin_viewer_1d_renderer_clear())
    assert receipt.cleared
    assert not render._trace_history_by_identity
    assert not render.curve.listDataItems()
    assert not render._pinned_trace_by_id
    assert render.waterfall.image.image is None
    assert render.waterfall.canvas.raw_image.size == 0
    rendered.clear()
    del single, trace, mode, selected, selected_by_context, current_only, overlay, toggled, waterfall, inherited, borrow, waterfall_state, payloads, base, view, invalid, refused, curve, item
    assert controller.acknowledge_viewer_1d_renderer_clear(receipt)
    assert controller.close_viewer_1d()

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
    single_refused = _shell(controller, ScientificPreferences(plot_mode="Single"))
    assert single_refused.scientific.traces == ()
    assert "conflicting units" in single_refused.scientific.status
    assert controller.viewer_1d_context.state.value == "ready"
    _reconcile_real_viewer(render, single_refused.scientific, controller.navigation)
    assert not render.curve.listDataItems()
    _release_real_viewer(controller, render)


def test_viewer_1d_cross_context_switch_and_workspace_close_are_positive(tmp_path, caplog) -> None:
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
    clear_results = []
    clear_allowed = False
    def clear_viewer(request):
        clear_results.append(clear_allowed)
        return viewer._new_viewer_1d_renderer_clear_receipt(request, clear_allowed)
    processed_browser = ProcessedBrowserOwner(
        save_path="",
        processing_mode="1D Viewer",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
    )
    assert processed_browser.begin_close()
    workspace = SimpleNamespace(_context_controller=controller,
        _shell=SimpleNamespace(browser=SimpleNamespace(
            reconcile_detector_mode=lambda _projection: None,
            reconcile_heavy_residency=lambda *_args, **_kwargs: None), scientific=SimpleNamespace(
            clear_viewer_1d=clear_viewer,
            expect_display_background=lambda _key: None,
            bottom_waterfall_active=False,
            drop_viewer_loading_snapshot=lambda: None,
            clear_workspace=lambda: True,
        )),
        _last_scientific_projection=object(), _preferences=ScientificPreferences(),
        _batch_terminal=BatchTerminalPresentationController(),
        _terminal_close=None, _closing=True, _lifecycle=SimpleNamespace(phase=RunPhase.IDLE),
        _close_identity=None, _clear_viewer_2d_renderer=lambda *, close: True,
        _processed_browser=processed_browser,
        _background_owner=SimpleNamespace(
            active_key=None, projection=lambda: None,
        ),
        _workspace_operations=SimpleNamespace(average_pending=None),
        _external_tools=SimpleNamespace(project=lambda **_kwargs: unavailable_external_tools()),
        _qualify_external_nexus=lambda **_kwargs: None,
    )
    workspace._clear_viewer_1d_renderer = partial(ScatteringWorkspace._clear_viewer_1d_renderer, workspace)
    workspace.__dict__.update(_closed=False, _sync_detector_demand=lambda: None, _retain_outgoing_display=False, _source_selection=SimpleNamespace(observation=None),
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(thaw=lambda: SimpleNamespace(
            processing_mode="1D Viewer", live_mode=False, source_spec=None, run_options={}))), _project_controls=lambda _snapshot: None,
        _start_permitted=lambda: (True, ""), _mutating_operation_busy=lambda: False,
        _context_projection=SimpleNamespace(
            build_shell=lambda **_: make_shell_projection(frame_count=0)
        ),
        _shell_revision=0, _run_executor=None, _notice_text="",
        _controls_readiness=None, _progress=ProgressProjection(),
            _browse_1d_release_debt=None, _scientific_repaint_pending=False,
            _ensure_timer=lambda: None, _release_browse_1d_debt=lambda: True,
            _retire_batch_presentation=lambda **_kwargs: None)
    workspace._lifecycle.__dict__.update(reset_permitted=False, active_run_identity=None, attempt_run_identity=None)
    notices = []; workspace._notice = notices.append
    def fail_render(*_args, **_kwargs):
        notices.append("injected-render-failure")
        raise RuntimeError("render")
    workspace._shell.apply_state = fail_render
    ScatteringWorkspace._refresh_shell(workspace)
    assert controller.viewer_1d_cleanup_pending and workspace._last_scientific_projection is None and "Passive shell render failed" in notices[-1]
    assert notices == ["injected-render-failure", "Passive shell render failed: render"]
    assert clear_results == [False]
    clear_allowed = True
    # Pytest retains logging traceback frames, including the exact projected
    # arrays.  The lease must refuse release until that test-owned alias drops.
    assert workspace._clear_viewer_1d_renderer(close=True) is False
    assert controller._viewer_1d.holder is not None
    logged = [record for record in caplog.records
              if record.getMessage() == "Passive shell render failed"]
    assert len(logged) == 1 and logged[0].exc_info[0] is RuntimeError
    logged[0].exc_info = None
    logged[0].exc_text = None
    import gc
    gc.collect()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not workspace._clear_viewer_1d_renderer(close=True):
        time.sleep(0.005)
    assert controller.viewer_1d_context is None
    controller.open_viewer_1d((str(one_d),)); _await_ready(controller)
    clear_allowed = False
    pending = ScatteringWorkspace.close_workspace(workspace)
    assert pending.cleanup_status.value == "cleanup_pending" and controller.viewer_1d_cleanup_pending
    clear_allowed = True
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not ScatteringWorkspace._edit_scientific_preference(
            workspace, ShellCommand(ShellCommandKind.CLEAR_1D)):
        time.sleep(0.005)
    assert controller.viewer_1d_context is None
    assert clear_results == [False, True, False, True]


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
        _workspace_operations=WorkspaceOperationOwner(),
        _metadata_operations=MetadataOperationOwner(),
        _analysis_operation_busy=lambda: False,
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
                currentRow=lambda: 0,
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


def test_viewer_1d_clicked_current_keeps_explicit_selection_in_both_modes(
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
        _workspace_operations=WorkspaceOperationOwner(),
        _metadata_operations=MetadataOperationOwner(),
        _analysis_operation_busy=lambda: False,
        _experiment_operation_busy=lambda: False,
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
        assert tuple(trace.frame for trace in single.scientific.traces) == frames

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


def test_viewer_1d_single_keeps_membership_without_preclear(
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

    assert calls == ["retire"]
    assert page._preferences.plot_mode == "Single"


def test_viewer_1d_page_commands_reload_status_and_native_restore(
    monkeypatch, tmp_path,
) -> None:
    """Real chooser/reload and clear receipts preserve the viewer boundary."""
    from pyqtgraph.Qt import QtWidgets

    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xrd_tools.session.intent_store import RunIntentStore

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = tuple(
        str(_write_xye(
            tmp_path / f"curve_{index}.xye", [0, 1], [index, index + 1],
        ))
        for index in (1, 2)
    )
    chooser_starts: list[str] = []
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            processing_mode="1D Viewer", project_root=str(tmp_path),
        )),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        viewer_1d_file_chooser=lambda start: (
            chooser_starts.append(start) or paths
        ),
    )

    def settle_ready() -> None:
        _await_ready(page._context_controller)
        page._refresh_shell()
        app.processEvents()

    try:
        page._run_action()
        settle_ready()
        controller = page._context_controller
        context = controller.viewer_1d_context
        assert context is not None and context.paths == paths
        assert chooser_starts == [""]
        owner = controller._viewer_1d
        assert owner.holder is not None and owner.holder.borrow is not None

        real_clear = page._shell.scientific.clear_viewer_1d
        receipts = []

        def observed_clear(request, **kwargs):
            receipt = real_clear(request, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(
            page._shell.scientific, "clear_viewer_1d", observed_clear,
        )
        page._run_action()
        settle_ready()
        assert chooser_starts == [""]
        assert len(receipts) == 1 and receipts[0].cleared
        assert controller.viewer_1d_context is not context
        assert controller.viewer_1d_context.paths == paths

        def refused_clear(_request, **_kwargs):
            raise RuntimeError("renderer clear refused")

        monkeypatch.setattr(
            page._shell.scientific, "clear_viewer_1d", refused_clear,
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PROCESSING_MODE, "Int 2D",
        ))
        assert page._intents.snapshot().thaw().processing_mode == "1D Viewer"
        assert controller.viewer_1d_cleanup_pending

        monkeypatch.setattr(
            page._shell.scientific, "clear_viewer_1d", real_clear,
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PROCESSING_MODE, "Int 2D",
        ))
        assert page._intents.snapshot().thaw().processing_mode == "Int 2D"
        assert controller.viewer_1d_context is None
        assert page._shell.scientific._processing_mode == "Int 2D"
    finally:
        for _ in range(100):
            receipt = page.close_workspace()
            if receipt.cleanup_status.value == "cleaned":
                break
            app.processEvents()
            time.sleep(0.005)
        assert receipt.cleanup_status.value == "cleaned"
        page.deleteLater()
        app.processEvents()

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
                "Thread": 0, "ThreadPoolExecutor": 0, "Queue": 0, "deque": 2}
    assert {name: identifiers.count(name) for name in expected} == expected
    calls = [(node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
             for node in nodes if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))]
    assert (calls.count("_new_viewer_1d_renderer_clear_request"), calls.count("Viewer1DRendererClearRequest"),
            calls.count("_new_viewer_1d_renderer_clear_receipt"), calls.count("Viewer1DRendererClearReceipt")) == (1, 0, 1, 0)
    for relative in ("io/viewer_1d.py", "session/viewer_1d.py"):
        source = (_ROOT / "src/xrd_tools" / relative).read_text()
        assert "from xdart" not in source and "import xdart" not in source
