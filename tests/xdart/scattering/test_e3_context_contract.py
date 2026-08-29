"""Frozen E3 context/controller ownership oracle.

The cases in this module intentionally exercise the controller through small
typed ports.  Executor construction itself is pinned separately below: the
canonical acquisition context must adopt the exact scan, configuration and
run-level display owner which the worker already created.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_values import (
    StandardDisplayPayload,
    StandardEventKind,
)
from xdart.gui.tabs.scattering.display_retirement import (
    NO_DISPLAY_RETIREMENT,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    DurableFinal,
    DurablePaused,
    ExecutorAccepted,
    ExecutorStartFailed,
    ExecutionEnded,
    OwnersClosed,
    PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import SlicePin
from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplaySelection,
    new_context_token,
)
from xdart.modules.frame_publication import FramePublication, PublicationStore
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io import Browse1DCache, FrameScalarCatalog, FrameScalarRow
from xrd_tools.io.output_transaction import TargetSnapshot
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


def _api():
    controller = import_module(
        "xdart.gui.tabs.scattering.context_controller"
    )
    projection = import_module(
        "xdart.gui.tabs.scattering.context_projection"
    )
    browse = import_module("xdart.gui.tabs.scattering.browse_values")
    return controller, projection, browse


def _configuration(*, gi: bool = False):
    values = {
        "source_spec": image_series_spec(Path("/data/a_0001.tif")),
        "poni_file": "/data/a.poni",
        "save_path": "/out/a.nxs",
        "output_mode": "Overwrite",
    }
    if gi:
        values["gi"] = {
            "enabled": True,
            "incidence_motor": "samth",
            "mode_1d": "qip",
            "mode_2d": "qip_qoop",
        }
    return RunIntent(**values).freeze()


def _view(label: int, value: float, *, gi: bool = False) -> FrameView:
    raw = np.full((2, 3), value)
    return FrameView(
        label,
        axis_1d=Axis("q", "A^-1", values=np.array([0.0, 1.0])),
        intensity_1d=np.array([value, value + 1.0]),
        axis_2d_x=Axis("q", "A^-1", values=np.arange(3.0)),
        axis_2d_y=Axis(
            "qoop" if gi else "chi",
            "A^-1" if gi else "deg",
            values=np.arange(2.0),
        ),
        intensity_2d=raw + 10.0,
        raw=raw,
        thumbnail=raw,
        source_path=f"/data/{'gi' if gi else 'scan'}_{label}.tif",
        source_frame_index=label,
    )


def _display(
    identity: RunIdentity,
    *,
    artifact: str,
    scan_key: str,
    value: float,
    gi: bool = False,
) -> RunDisplayState:
    display = RunDisplayState(identity, max_payload_items=4)
    display.set_factories(FrameRecordStore, PublicationStore)
    display.configure(partition_count=1, npt=2, frame_bytes=48)
    owner = display.add_artifact(
        Path(artifact),
        scan_key,
        mask=None,
        mask_saturation=True,
        measurement_mode="GI" if gi else "Standard",
        gi_incidence_motor="samth" if gi else "",
        gi_resolved_motor="samth" if gi else "",
        gi_mode_1d="qip" if gi else "",
        gi_mode_2d="qip_qoop" if gi else "",
    )
    view = _view(1, value, gi=gi)
    record = FrameRecord.from_view(view)
    publication = FramePublication(
        view,
        record=record,
        source_identity=f"{view.source_path}#1",
        scan_key=scan_key,
    )
    key = display.append_navigation(
        scan_key, artifact, 1
    ).appended
    display.retain_frame(
        owner,
        key,
        record,
        publication,
        source_identity=publication.source_identity,
        frame_mask_qualified=False,
    )
    display.put_payload(
        StandardDisplayPayload(
            0,
            key,
            f"{'GI' if gi else 'Standard'} · {scan_key} · frame 1",
            view,
            measurement_mode="GI" if gi else "Standard",
            gi_incidence_motor="samth" if gi else "",
            gi_resolved_motor="samth" if gi else "",
            gi_mode_1d="qip" if gi else "",
            gi_mode_2d="qip_qoop" if gi else "",
        )
    )
    return display


def _acquisition(
    *,
    gi: bool = False,
    configuration=None,
    identity: RunIdentity | None = None,
):
    configuration = configuration or _configuration(gi=gi)
    identity = identity or RunIdentity.from_configuration(configuration)
    scan = object()
    display = _display(
        identity,
        artifact="/out/a.nxs",
        scan_key="run.a",
        value=1.0,
        gi=gi,
    )
    context = AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=configuration,
        config_generation=configuration.generation,
        config_fingerprint=configuration.fingerprint,
        run_scan_key="run.a",
        source_path="/data/a_0001.tif",
        scan=scan,
        frame=None,
        frame_ids=display.catalog,
        frames=display.artifacts,
        viewer_rows_1d=(),
        viewer_rows_2d=(),
        publication_store=display,
        origin="scattering-standard",
        poni_identity=configuration.poni_file,
    )
    context.adopt_record_store(display)
    return identity, context


def _browse(
    context_token: str,
    generation: int,
    *,
    scan_key: str = "browse.b",
    request=None,
    scalar_row: FrameScalarRow | None = None,
    axes_1d: tuple[tuple[str, str, str, bool], ...] = (),
):
    _, _, browse_values = _api()
    request = request or browse_values.BrowseLoadRequest(
        context_token, generation, f"/processed/{scan_key}.nxs"
    )
    records = FrameRecordStore(max_items=8)
    publications = PublicationStore(max_items=8)
    view = _view(1, 20.0)
    record = FrameRecord.from_view(view)
    records.upsert(record, source_identity="/processed/b.nxs#1", persisted=True)
    publications.upsert(
        FramePublication(
            view,
            record=record,
            source_identity="/processed/b.nxs#1",
            scan_key=scan_key,
        )
    )
    catalog_row = FrameScalarRow(1) if scalar_row is None else scalar_row
    catalog = FrameScalarCatalog(
        request.source_path,
        "entry",
        (catalog_row,),
        axes_1d=axes_1d,
    )
    context = BrowseContext(
        context_token=context_token,
        load_generation=generation,
        operation=request,
        requested_path=request.source_path,
        scan_key=scan_key,
        scan=object(),
        frame=None,
        frame_ids=catalog.labels,
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=publications,
        record_store=records,
        scalar_catalog=catalog,
        browse_1d_cache=Browse1DCache(1 << 20),
        target_entry=catalog.entry,
        loaded_labels=catalog.labels,
        target_snapshot=TargetSnapshot(True, 1, 1, 1, 1, "a" * 64),
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    return request, context


class _ExecutorPort:
    def __init__(self, identity: RunIdentity, context: AcquisitionContext):
        self.identity = identity
        self.context = context
        self.pauses: list[RunIdentity] = []
        self.resumes: list[RunIdentity] = []
        self.stops: list[RunIdentity] = []
        self.durable_generation = 0

    def acquisition_context(self, identity):
        return self.context if identity is self.identity else None

    def pause(self, identity):
        assert identity is self.identity
        self.pauses.append(identity)
        self.durable_generation += 1
        return DurablePaused(identity, self.durable_generation)

    def resume(self, identity):
        assert identity is self.identity
        self.resumes.append(identity)

    def stop(self, identity):
        assert identity is self.identity
        self.stops.append(identity)


class _BrowsePort:
    def __init__(self):
        self.request = None
        self.context = None
        self.outcome = None
        self.released: list[BrowseContext] = []
        self.cancelled = []

    def begin(self, request):
        self.request = request
        return request

    def complete(self, context):
        if self.request in self.cancelled:
            self.release_context(context)
            self.outcome = None
            return
        _, _, browse_values = _api()
        self.context = context
        self.outcome = browse_values.BrowseLoadOutcome(
            self.request,
            browse_values.BrowseLoadStatus.READY,
        )

    def poll(self, request):
        return self.outcome if request is self.request else None

    def owns_outcome(self, outcome):
        return outcome is self.outcome

    def owns_request(self, request):
        return request is self.request

    def context_for_outcome(self, outcome):
        return self.context if outcome is self.outcome else None

    def consume(self, outcome):
        if outcome is not self.outcome:
            return None
        value = self.context
        self.context = None
        return value

    def cancel(self, request):
        self.cancelled.append(request)
        _, _, browse_values = _api()
        return browse_values.BrowseCleanupReceipt(
            request, CleanupStatus.CLEANED
        )

    def release_context(self, context):
        cache = context.browse_1d_cache
        if cache is not None:
            context.invalidate()
            cache.close()
            context.detach_browse_1d_cache(cache)
        context.release()
        if context.record_store is not None:
            context.record_store.clear()
        self.released.append(context)
        _, _, browse_values = _api()
        return browse_values.BrowseCleanupReceipt(
            context.load_request,
            CleanupStatus.CLEANED,
        )

    def close(self, expected=None):
        if self.request is not None:
            self.cancel(self.request)
        _, _, browse_values = _api()
        return browse_values.BrowseCleanupReceipt(
            expected if expected is not None else self.request,
            CleanupStatus.CLEANED,
        )


def _running_controller(*, gi: bool = False):
    controller_api, projection_api, _ = _api()
    configuration = _configuration(gi=gi)
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    accepted = lifecycle.preflight_accepted(
        PreflightAccepted(request, configuration)
    )
    identity = accepted.run_identity
    assert identity is not None
    _, acquisition = _acquisition(
        gi=gi,
        configuration=configuration,
        identity=identity,
    )
    assert lifecycle.executor_accepted(ExecutorAccepted(identity))
    executor = _ExecutorPort(identity, acquisition)
    loader = _BrowsePort()
    controller = controller_api.ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=loader,
        projection=projection_api.ContextProjection(),
    )
    selection = controller.adopt_acquisition(identity)
    assert selection.names(acquisition)
    return controller, lifecycle, executor, loader, acquisition


def _cold_controller(*, lifecycle=None):
    controller_api, projection_api, _ = _api()
    lifecycle = lifecycle or ScatteringCoordinator()
    loader = _BrowsePort()
    controller = controller_api.ContextController(
        lifecycle=lifecycle,
        executor=object(),
        browse_loader=loader,
        projection=projection_api.ContextProjection(),
    )
    return controller, lifecycle, loader


def _select_browse(controller, loader, *, scan_key: str = "browse.b"):
    request = controller.begin_browse(f"/processed/{scan_key}.nxs")
    _, context = _browse(
        request.token,
        request.load_generation,
        scan_key=scan_key,
        request=request,
    )
    loader.complete(context)
    outcome = controller.poll_browse()
    assert outcome is not None
    assert controller.selection.names(context)
    return request, context


def _current_key(controller):
    key = controller.navigation.current
    assert key is not None
    return key


def test_cold_idle_browse_uses_a_projection_only_identity() -> None:
    controller, lifecycle, loader = _cold_controller()

    request, browse = _select_browse(
        controller,
        loader,
        scan_key="cold.idle",
    )

    assert lifecycle.phase.value == "idle"
    assert controller.run_identity is None
    assert controller.acquisition_context is None
    key = _current_key(controller)
    assert type(key.run_identity) is RunIdentity
    assert key.run_identity.fingerprint == request.token
    payload = controller.project(key)
    assert type(payload) is StandardDisplayPayload
    assert payload.frame_key is key
    assert controller.selection.names(browse)


def test_paused_browse_keeps_the_exact_acquisition_run_identity() -> None:
    controller, lifecycle, executor, loader, _ = _running_controller()
    controller.pause()

    _select_browse(controller, loader, scan_key="paused")

    assert lifecycle.phase.value == "paused"
    assert controller.run_identity is executor.identity
    assert _current_key(controller).run_identity is executor.identity


def test_clean_failed_browse_is_allowed_but_pending_failed_cleanup_is_not() -> None:
    lifecycle = ScatteringCoordinator()
    start = lifecycle.begin_start()
    accepted = lifecycle.preflight_accepted(
        PreflightAccepted(start.request_id, _configuration())
    )
    identity = accepted.run_identity
    assert identity is not None
    failed = lifecycle.executor_start_failed(
        ExecutorStartFailed(identity, CleanupStatus.CLEANED)
    )
    assert failed.phase.value == "failed"
    assert lifecycle.reset_permitted is False
    controller, _, loader = _cold_controller(lifecycle=lifecycle)

    with pytest.raises(RuntimeError):
        controller.begin_browse("/processed/pending-cleanup.nxs")
    assert loader.request is None

    closed = lifecycle.owners_closed(OwnersClosed(identity))
    assert closed.phase.value == "failed"
    assert lifecycle.reset_permitted is True
    _select_browse(controller, loader, scan_key="clean.failed")
    assert controller.run_identity is None
    assert controller.selection is not None


def test_running_browse_is_refused_before_loader_admission() -> None:
    controller, lifecycle, _, loader, _ = _running_controller()

    with pytest.raises(RuntimeError):
        controller.begin_browse("/processed/running.nxs")

    assert lifecycle.phase.value == "running"
    assert loader.request is None


def test_browse_started_paused_can_complete_after_run_becomes_idle() -> None:
    controller, lifecycle, executor, loader, _ = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/after-run.nxs")
    ended = lifecycle.execution_ended(ExecutionEnded(executor.identity))
    assert ended.phase.value == "finalizing"
    final = lifecycle.durable_final(DurableFinal(executor.identity))
    assert final.phase.value == "idle"
    _, browse = _browse(
        request.token,
        request.load_generation,
        scan_key="after-run",
        request=request,
    )
    loader.complete(browse)

    outcome = controller.poll_browse()

    assert outcome is not None
    assert outcome.request is request
    assert controller.selection.names(browse)


def test_failed_cold_browse_replacement_clears_held_projection_identity() -> None:
    controller, _, loader = _cold_controller()
    _, browse_b = _select_browse(controller, loader, scan_key="browse.b")
    selection_b = controller.selection
    navigation_b = controller.navigation
    identity_b = _current_key(controller).run_identity

    request_c = controller.begin_browse("/processed/browse.c.nxs")
    assert browse_b.released is True
    assert controller.selection is selection_b
    assert controller.navigation is navigation_b
    _, _, browse_values = _api()
    loader.context = None
    loader.outcome = browse_values.BrowseLoadOutcome(
        request_c,
        browse_values.BrowseLoadStatus.FAILED,
        "replacement failed",
    )

    outcome = controller.poll_browse()

    assert outcome is loader.outcome
    assert controller.run_identity is None
    assert controller.browse_context is None
    assert controller.selection is None
    assert controller.navigation.frames == ()
    _select_browse(controller, loader, scan_key="browse.d")
    assert _current_key(controller).run_identity is not identity_b


def test_cold_browse_retirement_retries_the_exact_context() -> None:
    controller, _, loader = _cold_controller()
    request, browse = _select_browse(controller, loader, scan_key="cold.retire")
    _, _, browse_values = _api()
    release = loader.release_context
    calls = []

    def fail_once(context):
        calls.append(context)
        if len(calls) == 1:
            return browse_values.BrowseCleanupReceipt(
                request,
                CleanupStatus.CLEANUP_PENDING,
            )
        return release(context)

    loader.release_context = fail_once

    assert controller.apply_display_retirement(NO_DISPLAY_RETIREMENT) is False
    assert controller.browse_context is browse
    assert browse.released is False
    assert controller.apply_display_retirement(NO_DISPLAY_RETIREMENT) is True
    assert calls == [browse, browse]
    assert browse.released is True
    assert controller.browse_context is None
    assert controller.selection is None
    assert controller.run_identity is None


def test_case_01_standard_reaches_exact_durable_pause_latch():
    controller, lifecycle, executor, _, acquisition = _running_controller()
    result = controller.pause()
    assert result.run_identity is executor.identity
    assert result.durable_generation == 1
    assert lifecycle.phase.value == "paused"
    assert controller.selection.names(acquisition)


def test_c1_pause_resume_is_an_exact_acquisition_pointer():
    controller, lifecycle, executor, _, acquisition = _running_controller()
    owner = acquisition.hydration_owner.as_tuple()
    controller.pause()
    selection = controller.resume()
    assert lifecycle.phase.value == "running"
    assert selection.names(acquisition)
    assert controller.acquisition_context is acquisition
    assert acquisition.hydration_owner.as_tuple() == owner
    assert executor.resumes == [executor.identity]


def test_case_02_gi_motor_axes_and_context_survive_pause_resume():
    controller, _, _, loader, acquisition = _running_controller(gi=True)
    acquisition_key = _current_key(controller)
    before = controller.project(acquisition_key)
    controller.pause()
    _select_browse(controller, loader)
    controller.resume()
    after = controller.project(acquisition_key)
    assert controller.acquisition_context is acquisition
    assert before.gi_resolved_motor == after.gi_resolved_motor == "samth"
    np.testing.assert_array_equal(
        before.view.axis_2d_y.values,
        after.view.axis_2d_y.values,
    )


def test_case_03_browse_renders_raw_cake_1d_title_and_provenance():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _, browse = _select_browse(controller, loader)
    browse.stamp_provenance(
        calibration="poni-b",
        mask="mask-b",
        result="result-b",
    )
    payload = controller.project(_current_key(controller))
    assert payload.view.raw is not None
    assert payload.view.intensity_2d is not None
    assert payload.view.intensity_1d is not None
    assert "browse.b" in payload.title
    assert (
        browse.calibration_identity,
        browse.mask_identity,
        browse.result_identity,
    ) == ("poni-b", "mask-b", "result-b")


def test_case_04_browse_never_mutates_acquisition_owners():
    controller, _, _, loader, acquisition = _running_controller()
    frozen = (
        acquisition,
        acquisition.run_configuration,
        acquisition.scan_key,
        acquisition.poni_identity,
        acquisition.record_store,
        acquisition.publication_store,
        acquisition.publication_store.catalog_snapshot(),
    )
    controller.pause()
    _select_browse(controller, loader)
    assert frozen == (
        acquisition,
        acquisition.run_configuration,
        acquisition.scan_key,
        acquisition.poni_identity,
        acquisition.record_store,
        acquisition.publication_store,
        acquisition.publication_store.catalog_snapshot(),
    )


def test_case_05_resume_selects_exact_a_without_restoration_writes():
    controller, _, executor, loader, acquisition = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    owner = acquisition.hydration_owner
    selection = controller.resume()
    assert selection.names(acquisition)
    assert controller.acquisition_context is acquisition
    assert acquisition.hydration_owner is not owner
    assert acquisition.hydration_owner.as_tuple() == owner.as_tuple()
    assert executor.resumes == [executor.identity]


def test_case_06_next_acquisition_frame_projects_from_exact_a():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    controller.resume()
    payload = controller.project(_current_key(controller))
    assert payload.view.raw[0, 0] == 1.0
    assert controller.selection.names(acquisition)


def test_case_07_resume_preserves_publication_and_retention_state():
    controller, _, _, loader, acquisition = _running_controller()
    store = acquisition.publication_store
    catalog = store.catalog_snapshot()
    residency = store.residency_snapshot()
    payloads = tuple(store.payloads.items())
    publications = tuple(
        (name, owner.publications.labels())
        for name, owner in store.artifacts.items()
    )
    controller.pause()
    _select_browse(controller, loader)
    controller.resume()
    assert acquisition.publication_store is store
    assert store.catalog_snapshot() == catalog
    assert store.residency_snapshot() == residency
    assert tuple(store.payloads.items()) == payloads
    assert tuple(
        (name, owner.publications.labels())
        for name, owner in store.artifacts.items()
    ) == publications


def test_case_08_delayed_browse_load_after_resume_is_inert():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/late.b.nxs")
    controller.resume()
    _, browse = _browse(
        request.token,
        request.load_generation,
        request=request,
    )
    loader.complete(browse)
    assert controller.poll_browse() is None
    assert controller.selection.names(acquisition)
    assert browse.released is True


def test_case_09_delayed_projection_after_resume_is_inert():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    request = controller.project_request(_current_key(controller))
    controller.resume()
    assert controller.resolve_projection(request) is None


def test_foreign_browse_generation_is_not_a_load_identity():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/browse.b.nxs")
    _, _, browse_values = _api()
    foreign = browse_values.BrowseLoadRequest(
        request.token,
        request.load_generation + 1,
        request.source_path,
    )
    loader.outcome = browse_values.BrowseLoadOutcome(
        foreign, browse_values.BrowseLoadStatus.READY
    )
    _, loader.context = _browse(
        foreign.token,
        foreign.load_generation,
        request=foreign,
    )
    assert controller.poll_browse() is None
    assert controller.selection.names(acquisition)
    assert controller.browse_context is None


def test_projection_rejects_stale_source_epoch_until_exact_readoption():
    controller, _, executor, _, acquisition = _running_controller()
    current = _current_key(controller)
    request = controller.project_request(current)
    assert len(controller.project_navigation()) == 1
    assert controller.resident_frame_keys == frozenset({current})
    acquisition.rescope_to("run.a.member.2", "/data/a_0002.tif")
    assert controller.resolve_projection(request) is None
    assert controller.project_navigation() == ()
    assert controller.resident_frame_keys == frozenset()
    selection = controller.adopt_acquisition(executor.identity)
    assert selection.owner == acquisition.hydration_owner
    # The context's run-level store is still the exact same owner; a new
    # member/artifact becomes projectable when its catalog entry arrives.
    assert controller.acquisition_context is acquisition


def test_pin_axis_reseed_projects_the_exact_unselected_owned_frame() -> None:
    controller, _, _, _, acquisition = _running_controller()
    display = acquisition.publication_store
    first = _current_key(controller)
    first_payload = controller.project(first)
    assert type(first_payload) is StandardDisplayPayload
    display.put_payload(replace(first_payload, wavelength_m=1.0e-10))
    initial_pin = SlicePin(first, "Q", 0.5, 1.0)
    initial_preferences = ScientificPreferences(
        plot_axis="Q",
        plot_mode="Overlay",
        slice_enabled=True,
        slice_center=0.5,
        slice_width=1.0,
        slice_pins=(initial_pin,),
    )
    initial_payloads = controller.project_navigation(
        preferences=initial_preferences,
    )
    assert tuple(payload.frame_key for payload in initial_payloads) == (
        first,
    )
    assert controller.commit_navigation_projection((first,))

    second_delta = display.append_navigation(
        "run.a",
        "/out/a.nxs",
        2,
    )
    second = second_delta.appended
    second_view = _view(2, 2.0)
    display.put_payload(StandardDisplayPayload(
        0,
        second,
        "Standard · run.a · frame 2",
        second_view,
        wavelength_m=1.0e-10,
    ))
    assert controller.accept_navigation(
        second_delta,
        plot_mode="Overlay",
        follow_latest=True,
    )
    assert controller.select_navigation(second, (second,))

    retargeted = replace(initial_pin, plot_axis="2theta")
    preferences = replace(
        initial_preferences,
        plot_axis="2theta",
        slice_pins=(retargeted,),
    )
    payloads = controller.project_navigation(preferences=preferences)
    assert tuple(payload.frame_key for payload in payloads) == (
        second,
        first,
    )

    projection = build_scientific_projection(
        payloads,
        controller.navigation,
        controller.resident_frame_keys,
        preferences,
        "",
    )
    assert projection.title == "scan_2.tif"
    assert len(projection.pinned_traces) == 1
    pinned = projection.pinned_traces[0]
    assert pinned.pin is retargeted
    assert pinned.pin.plot_axis == "2theta"
    assert pinned.trace.frame is first


def test_duplicate_context_ready_adoption_keeps_exact_display_generation():
    controller, _, executor, _, _ = _running_controller()
    selection = controller.selection
    assert selection is not None
    request = controller.project_request(_current_key(controller))

    duplicate = controller.adopt_acquisition(executor.identity)

    assert duplicate is selection
    assert controller.selection is selection
    assert duplicate.display_generation == selection.display_generation
    assert controller.resolve_projection(request) is not None


def test_navigation_holds_exact_current_until_follow_latest_is_reenabled():
    controller, _, _, _, acquisition = _running_controller()
    first = _current_key(controller)
    display = acquisition.publication_store

    second_delta = display.append_navigation(
        "run.a",
        "/out/a.nxs",
        2,
    )
    assert controller.accept_navigation(
        second_delta,
        follow_latest=False,
    )
    assert controller.navigation.frames == (first, second_delta.appended)
    assert controller.navigation.current is first
    assert controller.navigation.selected == (first,)

    assert controller.select_latest_navigation()
    assert controller.navigation.current is second_delta.appended
    assert controller.navigation.selected == (second_delta.appended,)

    third_delta = display.append_navigation(
        "run.a",
        "/out/a.nxs",
        3,
    )
    assert controller.accept_navigation(
        third_delta,
        follow_latest=True,
    )
    assert controller.navigation.current is third_delta.appended
    assert controller.navigation.selected == (third_delta.appended,)


def test_delayed_context_ready_preserves_valid_paused_browse_selection():
    controller, _, executor, loader, _ = _running_controller()
    controller.pause()
    _, browse = _select_browse(
        controller,
        loader,
        scan_key="paused.browse",
    )
    selection = controller.selection
    assert selection is not None
    request = controller.project_request(_current_key(controller))

    delayed = controller.adopt_acquisition(executor.identity)

    assert delayed is selection
    assert controller.selection is selection
    assert selection.names(browse)
    assert controller.resolve_projection(request) is not None


def test_delayed_context_ready_preserves_pending_browse_replacement():
    controller, _, executor, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader, scan_key="browse.b")
    selection = controller.selection
    navigation = controller.navigation
    assert selection is not None
    replacement = controller.begin_browse("/processed/browse.c.nxs")
    assert controller.selection is selection

    delayed = controller.adopt_acquisition(executor.identity)

    assert delayed is selection
    assert controller.selection is selection
    assert controller.navigation == navigation
    assert controller.browse_pending
    assert replacement is loader.request


def test_equal_but_distinct_selection_is_inert():
    controller, _, _, _, _ = _running_controller()
    request = controller.project_request(_current_key(controller))
    foreign = replace(request.selection)
    assert foreign == request.selection
    assert foreign is not request.selection
    from xdart.gui.tabs.scattering.context_projection import ProjectionRequest

    assert controller.resolve_projection(
        ProjectionRequest(
            request.run_identity,
            foreign,
            request.frame,
        )
    ) is None


def test_case_10_equal_frame_labels_never_cross_contexts():
    controller, _, _, loader, _ = _running_controller()
    a = controller.project(_current_key(controller))
    controller.pause()
    _select_browse(controller, loader)
    b = controller.project(_current_key(controller))
    assert a.view.raw[0, 0] == 1.0
    assert b.view.raw[0, 0] == 20.0
    assert a.view is not b.view
    assert controller.owns_frame(a.frame_key) is False
    with pytest.raises(RuntimeError, match="no display selection"):
        controller.project(a.frame_key)


def test_case_11_dotted_stems_keep_one_canonical_scan_key(tmp_path):
    from tests.xdart.scattering.test_e4_preview_transport import (
        _write_processed,
    )

    processed, _raw = _write_processed(tmp_path)
    dotted = processed.with_name("run.with.dots.nexus")
    processed.rename(dotted)
    _, _, browse_values = _api()
    request = browse_values.BrowseLoadRequest(
        "browse-token",
        1,
        str(dotted),
    )
    assert request.source_path.endswith("run.with.dots.nexus")
    assert browse_values.canonical_browse_scan_key(
        request.source_path
    ) == "run.with.dots"


def test_case_12_final_a_and_b_remain_independently_selectable():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    _, browse = _select_browse(controller, loader)
    assert controller.select_acquisition().names(acquisition)
    assert controller.select_browse().names(browse)
    assert (
        controller.project(_current_key(controller)).view.raw[0, 0]
        == 20.0
    )


def test_case_13_close_during_browse_load_cleans_b_only():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/pending.nxs")
    controller.close()
    assert loader.cancelled == [request]
    assert acquisition.record_store is not None
    assert acquisition.publication_store is not None


def test_case_14_replacement_releases_b_once_and_accepts_c_only():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _, browse_b = _select_browse(controller, loader, scan_key="browse.b")
    request_c = controller.begin_browse("/processed/browse.c.nxs")
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    loader.complete(browse_c)
    controller.poll_browse()
    assert loader.released.count(browse_b) == 1
    assert controller.browse_context is browse_c


def test_same_scan_name_replacement_rejects_old_context_projection():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader, scan_key="same.scan")
    old_projection = controller.project_request(
        _current_key(controller))
    request = controller.begin_browse("/other/same.scan.nxs")
    _, replacement = _browse(
        request.token,
        request.load_generation,
        scan_key="same.scan",
        request=request,
    )
    loader.complete(replacement)
    assert controller.poll_browse() is not None
    assert controller.resolve_projection(old_projection) is None


def test_case_15_stop_while_b_selected_commands_acquisition_a():
    controller, _, executor, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    result = controller.stop()
    assert result.run_identity is executor.identity
    assert executor.stops == [executor.identity]


def test_case_16_retention_is_exactly_one_a_and_at_most_one_b():
    controller, _, _, loader, acquisition = _running_controller()
    controller.pause()
    _, browse_b = _select_browse(controller, loader, scan_key="browse.b")
    request_c = controller.begin_browse("/processed/browse.c.nxs")
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    loader.complete(browse_c)
    controller.poll_browse()
    assert controller.retained_contexts == (acquisition, browse_c)
    assert browse_b.released is True


def test_executor_mapping_adopts_exact_worker_owners_without_proxy_or_copy():
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    executor = StandardRunExecutor()
    run = _StandardRun(
        configuration,
        identity,
        object(),
        None,
        None,
        None,
        Path(configuration.save_path),
    )
    display = run.display
    display.set_factories(FrameRecordStore, PublicationStore)
    display.configure(partition_count=1, npt=2, frame_bytes=48)
    owner = display.add_artifact(
        Path(configuration.save_path),
        "run.a",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )
    context = executor._adopt_acquisition_context(
        run,
        owner,
        source_path="/data/a_0001.tif",
    )
    executor._active = run
    assert context.scan is run.scan
    assert context.run_configuration is configuration
    assert context.record_store is display
    assert context.publication_store is display
    assert context.frames is display.artifacts
    assert executor.acquisition_context(identity) is context
    events = executor.drain_events()
    assert len(events) == 1
    assert events[0].kind is StandardEventKind.CONTEXT_READY
    assert events[0].run_identity is identity
    assert events[0].frame_key is None
    assert events[0].detail == ""
