"""Frozen E6 Browse A oracle: one admitted concrete hydration owner.

Browse A relocates the accepted E4 cold-B lifecycle without changing the
Browse decision/issuer.  The controller binds exactly one private owner at
admission for both warm (borrowed acquisition transport) and cold (owned
fallback transport) Browse contexts.  Browse B's one-read resolver remains a
separate, held packet.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import inspect
import threading
import time

import numpy as np

from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.session.hydration import (
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
)

from tests.xdart.scattering.test_e4_preview_transport import (
    _adopted_browse,
    _adopted_cold_browse,
    _browse_key,
    _demote_browse_publication,
    _instrument_reads,
    _transport_api,
    _write_processed,
)


def _warm_browse_without_detector(tmp_path):
    """Warm Browse (borrowed transport) over a 1-D-only container."""

    from tests.xdart.scattering.test_e3_context_contract import (
        _running_controller,
    )
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.context_controller import ContextController
    from xdart.gui.tabs.scattering.context_projection import ContextProjection

    processed, _raw = _write_processed(
        tmp_path, labels=(1, 2, 3), thumbnails=False
    )
    _, lifecycle, executor, _, acquisition = _running_controller()
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=BrowseLoader(max_items=32),
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(executor.identity)
    controller.pause()
    request = controller.begin_browse(str(processed))
    deadline = time.monotonic() + 15.0
    outcome = None
    while outcome is None and time.monotonic() < deadline:
        outcome = controller.poll_browse()
        if outcome is None:
            time.sleep(0.005)
    assert outcome is not None and outcome.request is request
    browse = controller.browse_context
    assert browse is not None and browse.loaded
    return controller, acquisition, browse, processed


def _settle_transport(transport, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while (
        transport.queued_token is not None
        or transport.active_token is not None
    ) and time.monotonic() < deadline:
        time.sleep(0.005)


def _admitted_browse_token(controller, browse, label):
    """The exact token the Browse resolver admits, from public values."""
    from xrd_tools.session.hydration import HydrationToken

    read_key = HydrationReadKey(
        HydrationScope(*browse.hydration_owner.as_tuple()),
        browse.requested_path,
        label,
        HydrationPurpose.PREVIEW,
    )
    return HydrationToken(
        read_key, controller.selection.display_generation
    )


def _bound_owner(controller):
    owner = controller._browse_hydration_owner
    owner_type = import_module(
        "xdart.gui.tabs.scattering.browse_hydration"
    )._BrowseHydrationOwner
    assert type(owner) is owner_type
    return owner


def test_owner_module_replaces_publication_store_attachment_architecture():
    hydration = import_module(
        "xdart.gui.tabs.scattering.browse_hydration"
    )
    preview = import_module("xdart.gui.tabs.scattering.browse_preview")

    assert hydration._BrowseHydrationOwner.__module__ == hydration.__name__
    source = inspect.getsource(preview)
    assert "_BROWSE_PREVIEW_OWNER" not in source
    assert "_xdart_browse_preview_owner" not in source
    assert "_browse_preview_owner" not in source
    assert "setattr(store" not in source
    assert "_BrowseHydrationOwner(" not in source
    assert "HydrationTransport" not in source
    assert "display_transport" not in source


def test_cold_browse_binds_owned_owner_before_projection_without_attachment(
    tmp_path,
):
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)

    assert owner.names(browse)
    assert owner.owns_transport is True
    assert owner.transport.worker is None
    assert not hasattr(browse.publication_store, "_xdart_browse_preview_owner")

    key = _browse_key(controller, 1)
    assert controller.project(key) is None
    assert controller._browse_hydration_owner is owner


def test_warm_browse_binds_borrowed_owner_before_projection(tmp_path):
    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)

    assert owner.names(browse)
    assert owner.owns_transport is False
    assert owner.transport is acquisition.publication_store.transport
    assert owner.transport._commit.__self__ is acquisition.publication_store
    assert owner.transport._completion_sink is None
    assert not hasattr(browse.publication_store, "_xdart_browse_preview_owner")


def test_equal_valued_foreign_browse_cannot_use_bound_owner_transport(
    monkeypatch,
    tmp_path,
):
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)
    key = _browse_key(controller, 1)
    foreign = replace(browse)
    assert foreign is not browse
    assert foreign.context_token == browse.context_token
    assert foreign.scan_key == browse.scan_key
    assert foreign.requested_path == browse.requested_path
    assert foreign.publication_store is browse.publication_store
    assert foreign.commit_gate is browse.commit_gate
    assert owner.names(foreign) is False

    entered = threading.Event()

    def forbidden_read(*_args, **_kwargs):
        entered.set()
        raise AssertionError("foreign Browse reached the bound transport")

    monkeypatch.setattr(_transport_api(), "read_frame_preview", forbidden_read)
    controller._runtime._browse = foreign
    try:
        assert controller.project(key) is None
        assert entered.wait(timeout=0.1) is False
        assert owner.transport.active_token is None
        assert owner.transport.queued_token is None
    finally:
        controller._runtime._browse = browse


def test_warm_browse_transport_choice_is_frozen_once_at_admission(
    monkeypatch,
    tmp_path,
):
    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    admitted_transport = owner.transport
    key = _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)
    calls: list[str] = []

    class _LateTransport:
        def submit(self, _request, **_kwargs):
            calls.append("late")
            return None

    monkeypatch.setattr(
        admitted_transport,
        "submit",
        lambda _request, **_kwargs: calls.append("admitted"),
    )
    monkeypatch.setattr(acquisition.publication_store, "_transport", _LateTransport())

    assert controller.project(key) is None
    assert calls == ["admitted"]
    assert owner.transport is admitted_transport


def test_warm_browse_never_retires_borrowed_acquisition_transport(
    monkeypatch,
    tmp_path,
):
    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    key = _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)
    entered = threading.Event()
    release = threading.Event()
    preview_module = import_module("xrd_tools.io.frame_preview")
    original = preview_module.read_frame_preview
    retire_calls = []

    def holding_read(read_key, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    monkeypatch.setattr(
        owner.transport,
        "retire",
        lambda **kwargs: retire_calls.append(kwargs) or True,
    )
    assert controller.project(key) is None
    assert entered.wait(timeout=10.0)

    receipt = controller.close()
    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller._browse_hydration_owner is owner
    assert retire_calls == []

    release.set()
    deadline = time.monotonic() + 10.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        receipt = controller.close()

    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert browse.released is True
    assert owner.transport is acquisition.publication_store.transport
    assert retire_calls == []
    assert controller._browse_hydration_owner is None


def test_pending_cleanup_retains_exact_owned_owner_until_clean(
    monkeypatch,
    tmp_path,
):
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)
    key = _browse_key(controller, 1)
    entered = threading.Event()
    release = threading.Event()
    preview_module = import_module("xrd_tools.io.frame_preview")
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    assert controller.project(key) is None
    assert entered.wait(timeout=10.0)

    pending = controller.close()
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller._browse_hydration_owner is owner
    assert owner.names(browse)
    assert browse.released is False

    release.set()
    deadline = time.monotonic() + 10.0
    final = pending
    while (
        final.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        final = controller.close()

    assert final.cleanup_status is CleanupStatus.CLEANED
    assert browse.released is True
    assert controller._browse_hydration_owner is None
    assert owner.transport.worker is None or not owner.transport.worker.is_alive()


def test_missing_controller_owner_fails_closed_without_loader_release(
    monkeypatch,
    tmp_path,
):
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)
    real_release = controller._browse_loader.release_context
    releases = []

    def observe_release(context):
        releases.append(context)
        return real_release(context)

    monkeypatch.setattr(
        controller._browse_loader,
        "release_context",
        observe_release,
    )
    controller._browse_hydration_owner = None
    pending = controller.close()

    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert releases == []
    assert browse.released is False

    controller._browse_hydration_owner = owner
    cleaned = controller.close()
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert releases == [browse]
    assert browse.released is True


def test_owned_owner_survives_retirement_and_one_loader_release_failure(
    monkeypatch,
    tmp_path,
):
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseCleanupReceipt,
    )

    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)
    real_release = controller._browse_loader.release_context
    real_retire = owner.transport.retire
    calls = 0
    order = []

    def observe_retire(**kwargs):
        order.append("retire")
        return real_retire(**kwargs)

    def fail_once(context):
        nonlocal calls
        assert context is browse
        order.append("release")
        calls += 1
        if calls == 1:
            browse.invalidate()
            return BrowseCleanupReceipt(
                browse.load_request,
                CleanupStatus.CLEANUP_PENDING,
            )
        return real_release(context)

    monkeypatch.setattr(owner.transport, "retire", observe_retire)
    monkeypatch.setattr(controller._browse_loader, "release_context", fail_once)

    first = controller.close()
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller._browse_hydration_owner is owner
    assert browse.released is False
    assert browse.invalidated is True
    assert owner.names(browse) is False
    assert owner.transport.worker is None
    assert order == ["retire", "release"]

    second = controller.close()
    assert second.cleanup_status is CleanupStatus.CLEANED
    assert calls == 2
    assert browse.released is True
    assert controller._browse_hydration_owner is None
    assert order == ["retire", "release", "retire", "release"]


def test_foreign_equal_context_is_inert_for_terminal_repaint_and_release(
    monkeypatch,
    tmp_path,
):
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    owner = _bound_owner(controller)
    foreign = replace(browse)
    read_key = HydrationReadKey(
        HydrationScope(*browse.hydration_owner.as_tuple()),
        browse.requested_path,
        1,
        HydrationPurpose.PREVIEW,
    )
    owner._terminalize(read_key)
    owner._repaints.put(None)
    assert owner.detector_outcome(browse, 1) is not None
    assert owner.detector_outcome(foreign, 1) is None

    controller._runtime._browse = foreign
    try:
        assert controller.poll_browse_preview() is False
    finally:
        controller._runtime._browse = browse
    assert owner.consume_repaint() is True

    releases = []
    real_release = controller._browse_loader.release_context
    monkeypatch.setattr(
        controller._browse_loader,
        "release_context",
        lambda context: releases.append(context) or real_release(context),
    )
    receipt = owner.release(controller._browse_loader, foreign)
    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert releases == []
    assert browse.released is False
    assert controller._browse_hydration_owner is owner


def test_late_old_repaint_cannot_wake_equal_valued_replacement(tmp_path):
    hydration = import_module("xdart.gui.tabs.scattering.browse_hydration")
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    old_owner = _bound_owner(controller)
    replacement = replace(browse)
    replacement_owner = hydration._BrowseHydrationOwner(replacement)
    old_owner._repaints.put(None)

    controller._runtime._browse = replacement
    controller._browse_hydration_owner = replacement_owner
    try:
        assert controller.poll_browse_preview() is False
        assert old_owner.consume_repaint() is True
        assert replacement_owner.consume_repaint() is False
    finally:
        replacement_owner.retire()
        controller._runtime._browse = browse
        controller._browse_hydration_owner = old_owner


def test_poll_browse_never_replaces_owner_while_prior_release_is_pending(
    monkeypatch,
    tmp_path,
):
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseCleanupReceipt,
        BrowseLoadStatus,
    )
    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.context_controller import ContextController
    from xdart.gui.tabs.scattering.context_projection import ContextProjection
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator

    prior_controller, prior, processed = _adopted_cold_browse(tmp_path)
    prior_owner = _bound_owner(prior_controller)
    controller = ContextController(
        lifecycle=ScatteringCoordinator(),
        executor=object(),
        browse_loader=BrowseLoader(max_items=1),
        projection=ContextProjection(),
    )
    request = controller.begin_browse(str(processed))

    # Recreate the defensive state: the prior presentation remained current
    # while the replacement loader completed.  Commit A must fail closed here
    # instead of overwriting the owner after an ignored PENDING receipt.
    controller._runtime._browse = prior
    controller._browse_hydration_owner = prior_owner
    monkeypatch.setattr(
        controller,
        "_release_browse",
        lambda context: BrowseCleanupReceipt(
            context.load_request,
            CleanupStatus.CLEANUP_PENDING,
        ),
    )
    deadline = time.monotonic() + 15.0
    outcome = None
    while outcome is None and time.monotonic() < deadline:
        outcome = controller._browse_loader.poll(request)
        if outcome is None:
            time.sleep(0.005)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    candidate = controller._browse_loader.context_for_outcome(outcome)
    assert candidate is not None and candidate is not prior

    observed = controller.poll_browse()
    while (
        controller._browse_loader.owns_request(request)
        and time.monotonic() < deadline
    ):
        controller.poll_browse()
        time.sleep(0.005)

    assert observed is None
    assert controller.browse_context is prior
    assert controller._browse_hydration_owner is prior_owner
    assert controller._browse_request is None
    assert request is not prior.load_request
    assert candidate.invalidated is True
    assert candidate.released is True
    assert controller._browse_loader.owns_request(request) is False


# Warm borrowed-transport terminal repaint (E6-PM2 root cause, 2026-08-04): the
# borrowed sink is the acquisition owner's, so Browse reads its PUBLIC snapshot.


def test_warm_borrowed_terminal_1d_completion_wakes_once_and_qualifies(
    monkeypatch,
    tmp_path,
):
    controller, acquisition, browse, processed = (
        _warm_browse_without_detector(tmp_path)
    )
    owner = _bound_owner(controller)
    transport = owner.transport
    assert owner.owns_transport is False
    assert transport is acquisition.publication_store.transport

    key = _browse_key(controller, 1)
    # Resident 1-D record with no detector payload: still needs a read.
    resident = browse.publication_store.get(1)
    assert resident is not None and resident.view.raw is None
    assert resident.view.thumbnail is None
    counts = _instrument_reads(monkeypatch, processed)
    assert controller.project(key) is None  # exact hydration starts
    _settle_transport(transport)
    assert counts["detector"] == 0

    # THE defect: the borrowed completion must produce exactly one repaint.
    assert controller.poll_browse_preview() is True
    assert controller.poll_browse_preview() is False

    complete = controller.project(key)
    assert complete is not None and complete.view.raw is None
    np.testing.assert_allclose(
        complete.view.intensity_1d, np.array([2.0, 3.0, 4.0])
    )


def test_warm_borrowed_terminal_read_is_not_reenqueued_and_polling_settles(
    tmp_path,
):
    controller, _acquisition, browse, _processed = (
        _warm_browse_without_detector(tmp_path)
    )
    owner = _bound_owner(controller)
    transport = owner.transport
    key = _browse_key(controller, 1)

    assert controller.project(key) is None
    assert controller.browse_preview_polling_needed is True
    _settle_transport(transport)
    hydrated = transport.counters()[HydrationOutcome.HYDRATED]

    assert controller.poll_browse_preview() is True
    assert controller.project(key) is not None

    # Terminal here: no re-enqueue, and the owner stops requesting ticks.
    time.sleep(0.05)
    assert transport.queued_token is None
    assert transport.active_token is None
    assert transport.counters()[HydrationOutcome.HYDRATED] == hydrated
    assert controller.browse_preview_polling_needed is False
    assert owner.detector_outcome(browse, 1) is not None


def test_foreign_or_superseded_completion_is_inert_for_browse_owner(
    monkeypatch,
    tmp_path,
):
    from xrd_tools.session.hydration import (
        HydrationCompletion,
        HydrationToken,
    )

    controller, _acquisition, browse, _processed = (
        _warm_browse_without_detector(tmp_path)
    )
    owner = _bound_owner(controller)
    key = _browse_key(controller, 1)
    exact = _admitted_browse_token(controller, browse, 1)
    assert controller.project(key) is None  # admit the exact borrowed read

    # Foreign acquisition, equal-valued foreign Browse frame, superseded
    # generation: none is this owner's current Browse completion.
    inert = (
        HydrationToken(
            HydrationReadKey(
                HydrationScope("acq-token", "acq-scan", "acq-source", 7),
                "/acquisition/other.nxs",
                1,
                HydrationPurpose.PREVIEW,
            ),
            exact.presentation_generation,
        ),
        HydrationToken(
            replace(exact.read_key, frame_identity=2),
            exact.presentation_generation,
        ),
        HydrationToken(exact.read_key, exact.presentation_generation + 1),
    )
    assert all(token != exact for token in inert)
    monkeypatch.setattr(
        owner.transport,
        "completions",
        lambda: tuple(
            HydrationCompletion(token, HydrationOutcome.FAILED)
            for token in inert
        ),
    )
    assert owner.consume_repaint() is False
    assert controller.poll_browse_preview() is False
    assert owner.detector_outcome(browse, 1) is None
    assert owner.detector_outcome(browse, 2) is None
