from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardEventKind, StandardRunEvent
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.metadata_operations import MetadataOperationOwner
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.processed_browser import ProcessedBrowserOwner
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_operations import (
    WorkspaceOperationOwner,
    WorkspaceRefreshEffect,
)
from xdart.modules.display_context import (ContextKind, Viewer2DCleanupState, Viewer2DRendererClearReceipt,
                                           Viewer2DRendererClearRequest, Viewer2DReceiptPhase, Viewer2DState)
from xrd_tools.session.hydration import HydrationCompletion, HydrationOutcome, HydrationScope, HydrationToken
from xrd_tools.io import viewer_2d as viewer_api
from tests.core import test_viewer_2d as a0
from tests.xdart.scattering.test_e3_context_contract import _acquisition
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
    return ContextController(lifecycle=SimpleNamespace(
        phase=RunPhase.IDLE, reset_permitted=False, active_run_identity=None, attempt_run_identity=None),
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
        assert not controller.acknowledge_viewer_2d_renderer_clear(Viewer2DRendererClearReceipt(forged, forged is not request))
        assert state == (controller.viewer_2d_context, controller.viewer_2d_frame, controller._viewer_2d.receipt, controller._viewer_2d.clear_request)
    assert controller.acknowledge_viewer_2d_renderer_clear(Viewer2DRendererClearReceipt(request, True))
def _mount_source(tmp_path, family):
    base = np.arange(12, dtype=np.uint16).reshape(3, 4)
    suffix = {
        "edf": ".edf",
        "tiff": ".tiff",
        "cbf": ".cbf",
        "raw": ".raw",
        "hdf": ".h5",
        "nexus": ".nxs",
        "processed": ".nexus",
        "csv": ".csv",
        "npy2": ".npy",
        "npy3": ".npy",
        "npz": ".npz",
    }.get(family, ".nxs")
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
            a0._processed_source(
                handle, 2, str(raw.resolve()), 0, "/entry/data/data",
            )
            a0._processed_thumbnail(handle, 7, thumb, vmin=0.0, vmax=255.0)
            a0._finalize_processed(handle)
        policy = viewer_api.Viewer2DFormatPolicy(
            source_root=str(tmp_path)
        )
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
        monkeypatch.setattr("xdart.gui.tabs.scattering.context_controller.Viewer2DFormatPolicy", lambda: policy)
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
        assert payload is not None and payload.view.raw is controller.viewer_2d_frame.array and np.array_equal(payload.view.raw, value)
        assert all(getattr(payload.view, name) is None for name in (
            "axis_1d", "intensity_1d", "sigma_1d", "axis_2d_x", "axis_2d_y", "intensity_2d", "sigma_2d", "thumbnail", "geometry"))
        ledger = viewer_api.viewer_2d_selected_ledger(owner.catalog, owner.frame.label)
        assert (owner.receipt.phase, owner.receipt.capacity, owner.receipt.reserved) == (Viewer2DReceiptPhase.FRAME_READY_A, ledger.budget, ledger.admission)
        kind = ({"processed": ("Processed raw", "Thumbnail preview"), "csv": ("CSV matrix",)}.get(family, ("NumPy array",) * len(expected) if family in {"npy2", "npy3", "npz"} else ("Raw detector",)))[index]
        assert payload.title == f"{path.name} · frame {index + 1} · {kind}"
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
    assert owner.frame is controller._runtime._viewer_2d_frame and owner.receipt.phase is Viewer2DReceiptPhase.FRAME_READY_A and controller.viewer_2d_context.state is Viewer2DState.READY
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
        return (owner.context, owner.catalog, owner.receipt, owner.frame, owner.request, owner.request_token,
                owner.loading, owner.diagnostic, owner.changed, owner.latest_label, controller.selection, controller.navigation, controller._runtime._display_generation)
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
    relative_paths = ("src/xrd_tools/session/readiness.py", *(module + name for name in ("context_runtime.py", "context_controller.py", "context_projection.py", "controls_projection.py", "run_mode_projection.py", "page.py", "scientific_view.py")))
    trees, calls, identifiers = [], Counter(), Counter()
    for path in relative_paths:
        tree, path_calls, path_identifiers = _ast_facts((root / path).read_text())
        trees.append(tree)
        calls.update(path_calls)
        identifiers.update(path_identifiers)
    classes = tuple(node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    owners = tuple(node for node in classes if node.name == "_TwoDViewerOwner")
    assert len(owners) == calls["_TwoDViewerOwner"] == 1
    assert {node.name for node in owners[0].body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")} == {
        "activate", "dispose", "commit", "complete"}
    assert calls["RLock"] == calls["HydrationTransport"] == 1
    assert not {
        name
        for name in (
            "Thread",
            "ThreadPoolExecutor",
            "Timer",
            "Queue",
            "Process",
        )
        if calls[name]
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
    @dataclass(frozen=True)
    class _ScientificProjection:
        heavy: object
        background_enabled: bool = True

    @dataclass(frozen=True)
    class _ShellProjection:
        scientific: _ScientificProjection
        external_tools: object = None

    calls: list[object] = []
    clear_request = object()
    identity = RunIdentity(2, "delayed")
    processed_browser = ProcessedBrowserOwner(
        save_path="",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
    )
    controller = SimpleNamespace(
        viewer_2d_owned=True, viewer_2d_frame=object(), run_identity=identity,
        selection=None,
        poll_viewer_2d=lambda: False, poll_browse_preview=lambda: False,
        browse_pending=False, adopt_acquisition=lambda value: calls.append(("adopt", value)),
        begin_viewer_2d_renderer_clear=lambda: clear_request,
        acknowledge_viewer_2d_renderer_clear=lambda receipt: calls.append(("ack", receipt)) or receipt == "positive",
        close_viewer_2d=lambda: calls.append("close") or True)
    scientific = SimpleNamespace(
        clear_viewer_2d=lambda _request: "forged",
        drop_viewer_loading_snapshot=lambda: calls.append("drop-snapshot"),
        expect_display_background=lambda _key: None,
        trace_row_count=0,
        bottom_waterfall_active=False,
    )
    page = SimpleNamespace(
        _closing=False, _closed=False, _context_controller=controller,
        _shell=SimpleNamespace(scientific=scientific),
        _retain_outgoing_display=False,
        _scientific_repaint_pending=False,
        _waterfall_candidate_count=0,
        _workspace_operations=WorkspaceOperationOwner(),
        _metadata_operations=MetadataOperationOwner(),
        _processed_browser=processed_browser,
        _pending_viewer_2d_path=None,
        _batch_terminal=SimpleNamespace(
            active=False,
            project_progress=lambda progress: progress,
        ),
        _analysis_operation_busy=lambda: False,
        _experiment_operation_busy=lambda: False,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(
            revision=0,
            thaw=lambda: SimpleNamespace(processing_mode="Int 2D"),
        )),
        _last_scientific_projection=object(),
        _select_scan=lambda value, **_kwargs: calls.append(value),
        _poll_admission=lambda: False,
        _settle_browse_1d_before_drain=lambda: True,
        _dispatch_deferred_metadata=lambda: WorkspaceRefreshEffect.NONE,
        _batch_ready_to_paint=lambda: None,
        _advance_authored_asset_confirmation=lambda: None,
        _run_executor=SimpleNamespace(drain_events=lambda: (StandardRunEvent(identity, StandardEventKind.CONTEXT_READY),)),
        _lifecycle=SimpleNamespace(active_run_identity=identity, attempt_run_identity=None),
        _refresh_shell=lambda: calls.append("refresh"), _polling_needed=lambda: True,
        _run_timer=SimpleNamespace(stop=lambda: calls.append("stop")))
    page._clear_presentation_targets = partial(ScatteringWorkspace._clear_presentation_targets, page)
    page._retire_lost_reintegrate_successor = partial(
        ScatteringWorkspace._retire_lost_reintegrate_successor, page)
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
    canonical = np.arange(4.0).reshape(2, 2)
    frame, context = controller.viewer_2d_frame, object()
    retained = _ScientificProjection(
        heavy=SimpleNamespace(raw=canonical, frame=frame),
    )
    controller.viewer_2d_context = context
    controller.__dict__.update(
        synchronize_acquisition_scope=lambda: None,
        capture_norm_aggregate_for_refresh=lambda: None,
        cancel_browse_slices=lambda: None,
        viewer_2d_cleanup_pending=False,
        browse_context=None,
        navigation=SimpleNamespace(current=None), project_navigation=lambda **_: (),
        resident_frame_keys=(), projectable_contexts=(), selection=None,
        norm_aggregate=None, viewer_2d_diagnostic="")
    page._lifecycle.phase, page._lifecycle.reset_permitted = RunPhase.IDLE, False
    page.__dict__.update(
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(thaw=lambda: SimpleNamespace(
            processing_mode="2D Viewer", live_mode=False, source_spec=None, run_options={}))),
        _project_controls=lambda _snapshot: None, _start_permitted=lambda: (True, ""), _sync_detector_demand=lambda: None,
        _mutating_operation_busy=lambda: False,
        _preferences=ScientificPreferences(), _retain_outgoing_display=False,
        _external_tools=SimpleNamespace(project=lambda **_kwargs: None),
        _qualify_external_nexus=lambda **_kwargs: None,
        _source_selection=SimpleNamespace(observation=None),
            _context_projection=SimpleNamespace(
                build_shell=lambda **_: _ShellProjection(retained),
            ),
            _background_owner=SimpleNamespace(projection=lambda: None, active_key=None),
            _shell_revision=0, _controls_readiness=None, _progress=None,
        _notice_text="")
    def fail_render(_projection, *, preserve_display=False):
        calls.append("render-failed")
        raise RuntimeError("render")
    page._shell = SimpleNamespace(scientific=scientific, browser=SimpleNamespace(
        reconcile_detector_mode=lambda _projection: None,
        reconcile_heavy_residency=lambda *_args, **_kwargs: None), apply_state=fail_render)
    page._notice = lambda text: setattr(page, "_notice_text", text)
    page._last_scientific_projection = retained
    owner_state = (controller.viewer_2d_frame, controller.viewer_2d_context)
    assert retained.heavy.raw is canonical
    ScatteringWorkspace._refresh_shell(page)
    assert calls[-2:] == ["render-failed", "drop-snapshot"]
    assert page._notice_text == "Passive shell render failed: render"
    assert page._last_scientific_projection is None and owner_state == (controller.viewer_2d_frame, controller.viewer_2d_context)
    assert processed_browser.begin_close()

def test_real_viewer_chooser_preserves_opaque_identity_and_acquisition_isolation(monkeypatch) -> None:
    calls = []
    selected = "/opaque/data.npz::nested/image.npy"
    def forbidden(*_args, **_kwargs):
        raise AssertionError("viewer chooser entered an acquisition seam")
    for name in (
        "is_single_image_spec",
        "browse_start_dir",
        "remember_browse_path",
    ):
        monkeypatch.setattr(f"xdart.gui.tabs.scattering.page.{name}", forbidden)
    filesystem = SimpleNamespace(path=SimpleNamespace(**dict.fromkeys(
        ("abspath", "expanduser", "exists", "isfile", "realpath", "splitext"), forbidden)), stat=forbidden, open=forbidden)
    monkeypatch.setattr("xdart.gui.tabs.scattering.page.os", filesystem)
    controller = SimpleNamespace(
        viewer_2d_context=None, run_identity=None,
        open_viewer_2d=lambda path: calls.append(("open", path)) or object())
    intent = SimpleNamespace(
        processing_mode="2D Viewer",
        live_mode=False,
        run_options={},
    )
    page = SimpleNamespace(
        _context_controller=controller, _lifecycle=SimpleNamespace(phase=RunPhase.IDLE),
        _viewer_2d_start_directory=lambda: calls.append("start") or "/viewer",
        _viewer_2d_file_chooser=lambda start: calls.append(("choose", start)) or selected,
        _clear_viewer_2d_renderer=forbidden,
        _intents=SimpleNamespace(snapshot=lambda: SimpleNamespace(thaw=lambda: intent), commit=forbidden),
        _source_selection=SimpleNamespace(
            mode="directory",
            history={"directory": object()},
            live_source=object(),
        ),
        _workspace_operations=SimpleNamespace(
            average_pending=None,
            average_identity=None,
        ),
        _retire_batch_presentation=lambda: None,
        _retain_outgoing_display=False,
        _notice=lambda value: calls.append(("notice", value)),
        _ensure_timer=lambda: calls.append("timer"), _error_notice=forbidden,
        **dict.fromkeys(("select_source", "_queue_live_source_refresh", "_begin_run"), forbidden),
    )
    page._choose_viewer_2d_file = partial(ScatteringWorkspace._choose_viewer_2d_file, page)
    page._open_viewer_2d_path = partial(
        ScatteringWorkspace._open_viewer_2d_path,
        page,
    )
    source_state = (
        page._source_selection.mode,
        dict(page._source_selection.history),
        page._source_selection.live_source,
    )
    ScatteringWorkspace._run_action(page)
    assert calls == ["start", ("choose", "/viewer"), ("open", selected), ("notice", ""), "timer"]
    assert source_state == (
        page._source_selection.mode,
        page._source_selection.history,
        page._source_selection.live_source,
    )
    controller.viewer_2d_context = SimpleNamespace(original_path=selected)
    page._viewer_2d_file_chooser = forbidden
    def clear_renderer(*, close=False, preserve_navigation=False):
        assert close is True and preserve_navigation is False
        calls.append("clear")
        return True
    page._clear_viewer_2d_renderer = clear_renderer
    ScatteringWorkspace._run_action(page)
    assert calls[-4:] == ["clear", ("open", selected), ("notice", ""), "timer"]
def _scientific(monkeypatch, *, failing=False):
    """Observe real widget calls; only render/clear failures are injected."""
    view = ScientificView()
    events = []
    view.events = events

    def observe(target, name, method, action):
        original = getattr(target, method)
        def call(*args, **kwargs):
            events.append((name, action, *args))
            return original(*args, **kwargs)
        monkeypatch.setattr(target, method, call)

    for name, widget in (("raw", view.raw), ("splitter", view.image_splitter)):
        observe(widget, name, "hide", "hide")
        observe(widget, name, "setVisible", "visible")
    for name in ("title", "status", "progress"):
        observe(getattr(view, name), name, "setText", "text")
    for name in ("previous_frame", "next_frame"):
        observe(getattr(view, name), name, "setEnabled", "enabled")
    observe(view.frame_selector, "selector", "setCurrentIndex", "index")
    observe(view.vertical_splitter, "vertical", "setSizes", "sizes")
    for name, pane in (("raw", view.raw), ("cake", view.cake)):
        pane.render(np.arange(6.0).reshape(2, 3))
        image = pane.canvas.imageItem
        for method, action in (("prepareGeometryChange", "geometry"),
                               ("informViewBoundsChanged", "bounds"), ("update", "update")):
            observe(image, name, method, action)
        observe(image.pos_label, f"{name}-hover", "setText", "text")
        observe(pane.canvas.imageViewBox, f"{name}-range", "setRange", "range")
    view.curve.plot([1, 2], [3, 4])
    view.waterfall.render(np.ones((2, 3)))
    for name, value in dict(
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
        _rendered_cake_y_axis=object(), _rendered_trace_axis_key="Q",
    ).items():
        setattr(view, name, value)
    view.raw.fail = failing
    render = view.raw.render
    def observed_render(value, **kwargs):
        events.append(("raw", "render", value))
        if view.raw.fail:
            raise view.raw.fail if isinstance(view.raw.fail, BaseException) else ValueError("secret")
        view.raw.render_values = kwargs
        return render(value, **kwargs)
    monkeypatch.setattr(view.raw, "render", observed_render)
    return view

def _assert_neutral(view, title, status, *, known_empty=True) -> None:
    assert view._viewer_2d_known_empty is known_empty
    assert view._viewer_2d_payload is None
    assert view._frame_keys == view._selected_keys == view._rendered_trace_keys == ()
    assert view._label_indices == view._trace_history_by_identity == view._pinned_trace_by_id == {}
    assert view._heavy_available == frozenset()
    assert view.curve.listDataItems() == [] and view.waterfall.canvas.imageItem.image is None
    assert view._trace_selection_keys == view._trace_history_keys == view._waterfall_y_values == view._waterfall_source_keys == ()
    assert view._trace_history_scope is view._pinned_trace_scope is view._rendered_plot_options is None
    assert (view._rendered_plot_mode, view._bottom_waterfall_active) == ("", False)
    assert (view._rendered_overlay_step, view._waterfall_render_contract, view._rendered_image_axis,
            view._rendered_cake_axis_key, view._rendered_cake_x_axis, view._rendered_cake_y_axis,
            view._rendered_trace_axis_key) == (None,) * 7
    for pane in (view.raw, view.cake):
        assert pane.canvas.raw_image.size == pane.canvas.displayed_image.size == 0
        assert pane.canvas.imageItem.image is None
        np.testing.assert_array_equal(pane.canvas.imageItem.levels, (0.0, 1.0))
        assert all(getattr(pane.canvas.imageItem, name) is None for name in (
            "qimage", "_defferedLevels", "_displayBuffer", "_processingBuffer", "_imageNanLocations", "_imageHasNans"))
        assert pane.canvas.imageItem._lastDownsample == (1, 1)
        assert pane.canvas._level_cache is pane.canvas._level_scan_token is None
        assert pane.canvas.histogram.levels() == (0.0, 1.0)
        assert pane.canvas.histogram.lo_lim is pane.canvas.histogram.hi_lim is None
        assert pane.canvas.imageItem.transform().isIdentity()
        name = "raw-range" if pane is view.raw else "cake-range"
        rect = next(event[2] for event in reversed(view.events)
                    if event[:2] == (name, "range") and len(event) > 2)
        assert (rect.x(), rect.y(), rect.width(), rect.height()) == (0.0, 0.0, 1.0, 1.0)
    assert {"geometry", "bounds", "update"} <= {event[1] for event in view.events}
    assert view.frame_selector.count() == 0 and view.frame_selector.currentIndex() == -1
    assert not view.previous_frame.isEnabled() and not view.next_frame.isEnabled()
    assert (view.title.text(), view.status.text(), view.progress.text()) == (title, status, "0/0")
    assert all(item.isHidden() for item in (
        view.image_splitter, view.raw, view.cake, view.vertical_splitter.widget(1)))

def test_viewer_transaction_hides_renders_reveals_last_and_retries_same_array(monkeypatch) -> None:
    view = _scientific(monkeypatch, failing=True)
    array = np.arange(12.0).reshape(3, 4)
    identity = RunIdentity(1, "viewer-token")
    prior = DisplayFrameKey(identity, "viewer-2d", "viewer-2d", 2, 1)
    frame = DisplayFrameKey(identity, "viewer-2d", "viewer-2d", 7, 2)
    state = SimpleNamespace(
        processing_mode="2D Viewer", heavy=SimpleNamespace(frame=frame, raw=array, detector_shape=None),
        heavy_available=frozenset((frame,)), color_map="plasma", log_scale=True,
        title="image.npy · frame 2 · NumPy array", status="2D Viewer · NumPy array",
        background_set=False, background_enabled=True)
    navigation = SimpleNamespace(frames=(prior, frame), current=frame, selected=(frame,))
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
    _assert_neutral(
        view,
        "2D Viewer · Render failed",
        "2D Viewer · Render failed; retry available.",
        known_empty=False,
    )
    view.raw.clear = clear
    view.raw.fail = False
    view.events.clear()
    reconcile()
    assert view._viewer_2d_payload is array
    np.testing.assert_array_equal(view.raw.canvas.raw_image, array.T[:, ::-1])
    assert np.shares_memory(view.raw.canvas.raw_image, array)
    assert view.raw.render_values["level_scan_token"] == (id(frame), id(array))
    assert (view.raw.render_values["color_map"], view.raw.render_values["log_scale"]) == ("plasma", True)
    displayed = np.log10(array.T[:, ::-1] + 1.0)
    np.testing.assert_allclose(view.raw.canvas.displayed_image, displayed)
    np.testing.assert_allclose(view.raw.canvas.imageItem.image, displayed)
    expected_levels = np.percentile(displayed, (0.1, 99.9))
    np.testing.assert_allclose(view.raw.canvas.imageItem.levels, expected_levels)
    np.testing.assert_allclose(view.raw.canvas.histogram.levels(), expected_levels)
    assert (view.raw.canvas.histogram.lo_lim, view.raw.canvas.histogram.hi_lim) == (0.0, np.log10(12.0))
    assert view.raw.canvas.imageItem.mapRectToParent(view.raw.canvas.imageItem.boundingRect()) == QtCore.QRectF(0, 0, 3, 2)
    assert view.progress.text() == "2/2"
    assert [view.frame_selector.itemText(index) for index in range(view.frame_selector.count())] == ["1", "2"]
    assert view.frame_selector.itemData(0) is prior
    assert view.frame_selector.itemData(1) is frame
    assert view.cake.canvas.imageItem.image is view.waterfall.canvas.imageItem.image is None
    assert view.curve.listDataItems() == []
    assert view.norm.isHidden() and not view.background.isHidden()
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
    assert view.raw.canvas.imageItem.pos_label.text == ""
    assert view.previous_frame.isEnabled() and not view.next_frame.isEnabled()
    forged = DisplayFrameKey(RunIdentity(9, "native"), "viewer-2d", "viewer-2d", 99, 1)
    commands = []
    view.commandRequested.connect(commands.append)
    for selected, kind in ((navigation.current, ShellCommandKind.SELECT_FRAME),
                           (forged, ShellCommandKind.HYDRATE_FRAME),
                           (replace(navigation.current), ShellCommandKind.HYDRATE_FRAME)):
        commands.clear()
        blocker = QtCore.QSignalBlocker(view.frame_selector)
        view.frame_selector.clear()
        view.frame_selector.add_frame(str(selected.local_frame_label), selected, "")
        del blocker
        ScientificView._frame_selected(view, 0)
        assert commands[0].kind is kind
    canonical = view._viewer_2d_payload
    request = Viewer2DRendererClearRequest("viewer-token", 3, "0" * 64, 0)
    receipt = ScientificView.clear_viewer_2d(view, request)
    assert receipt.request is request and receipt.cleared
    assert canonical is array and view.raw.canvas.imageItem.image is None
    _assert_neutral(view, "Current", "")
    view.events.clear()
    reconcile()
    render_at = next(i for i, event in enumerate(view.events)
                     if event[:2] == ("raw", "render"))
    assert not any(event[:2] == ("splitter", "hide")
                   for event in view.events[:render_at])
    assert not any(event[:2] == ("raw", "hide")
                   for event in view.events[:render_at])
    assert not view._viewer_2d_known_empty
    interrupt = _scientific(monkeypatch)
    interrupt.raw.fail = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        reconcile(interrupt)

def test_processing_layout_rederives_viewer_and_both_native_rows_non_gui(monkeypatch) -> None:
    view = _scientific(monkeypatch)
    panes = (view.image_splitter, view.raw, view.cake, view.vertical_splitter.widget(1))
    initial_sizes = view.vertical_splitter.sizes()
    ScientificView._apply_processing_layout(view, "2D Viewer")
    assert not view.image_splitter.isHidden() and not view.raw.isHidden()
    assert view.cake.isHidden() and view.vertical_splitter.widget(1).isHidden()
    ScientificView._apply_processing_layout(view, "Int 2D")
    assert not any(item.isHidden() for item in panes)
    assert view.vertical_splitter.sizes() == initial_sizes
    ScientificView._apply_processing_layout(view, "Int 1D")
    assert view.image_splitter.isHidden()
    assert not any(item.isHidden() for item in panes[1:])
    ScientificView._apply_processing_layout(view, "Int 2D")
    assert not any(item.isHidden() for item in panes)
    assert ("vertical", "sizes", [500, 500]) in view.events
