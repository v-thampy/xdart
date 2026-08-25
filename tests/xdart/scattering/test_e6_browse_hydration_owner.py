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
    HydrationCompletion,
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


def _assert_browse_hydration_retains_nearest_complete_window(
    controller,
    browse,
    owner,
) -> None:
    from xdart.gui.tabs.scattering.display_runtime import (
        publication_needs_hydration,
    )

    labels = tuple(range(1, 10))
    store = browse.publication_store
    store.set_max_heavy_items(3)
    assert store.complete_labels(labels) == frozenset((7, 8, 9))

    first = _browse_key(controller, 1)
    assert controller.project(first) is None
    _settle_transport(owner.transport)
    assert store.complete_labels(labels) == frozenset((1, 7, 8))
    first_payload = controller.project(first)
    assert first_payload is not None
    assert first_payload.view.intensity_2d is not None

    sixth = _browse_key(controller, 6)
    assert controller.project(sixth) is None
    _settle_transport(owner.transport)
    assert store.complete_labels(labels) == frozenset((6, 7, 8))
    assert publication_needs_hydration(store.get(7), None) is False
    seventh = _browse_key(controller, 7)
    payload = controller.project(seventh)
    assert payload is not None
    assert payload.view.intensity_2d is not None
    assert owner.transport.active_token is None
    assert owner.transport.queued_token is None


def test_cold_browse_hydration_retains_nearest_complete_window(tmp_path):
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path,
        labels=tuple(range(1, 10)),
        loader_max=9,
    )
    owner = _bound_owner(controller)
    _assert_browse_hydration_retains_nearest_complete_window(
        controller,
        browse,
        owner,
    )


def test_warm_browse_hydration_retains_nearest_complete_window(tmp_path):
    controller, _acquisition, browse, _processed = _adopted_browse(
        tmp_path,
        labels=tuple(range(1, 10)),
    )
    owner = _bound_owner(controller)
    assert owner.owns_transport is False
    _assert_browse_hydration_retains_nearest_complete_window(
        controller,
        browse,
        owner,
    )


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
        def _submit_admission(self, _request, **_kwargs):
            calls.append("late")
            return None

    # Borrowed Browse admits through the private ticket seam (E6-PM2), so the
    # frozen-choice fact is observed there; the asserted contract is unchanged.
    monkeypatch.setattr(
        admitted_transport,
        "_submit_admission",
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
        lambda context, **_purpose: BrowseCleanupReceipt(
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


# E6-PM2 ratified admission tickets (2026-08-04): the owner tracks transport
# ticket OBJECTS, never token values and never the diagnostic deque, so an
# equal-valued token from a later admission cannot consume a stale completion
# and a rotated-out completion cannot be lost.


def _foreign_request(browse, label, generation):
    """An admission for a DIFFERENT frame of the same Browse artifact."""

    from xdart.modules.display_context import HydrationRequest
    from xrd_tools.session.hydration import HydrationToken

    read_key = HydrationReadKey(
        HydrationScope(*browse.hydration_owner.as_tuple()),
        browse.requested_path,
        label,
        HydrationPurpose.PREVIEW,
    )
    return HydrationRequest(
        label,
        HydrationPurpose.PREVIEW,
        generation,
        browse.hydration_owner,
        (browse.publication_store,),
        browse.commit_gate,
        read_key=read_key,
        token=HydrationToken(read_key, generation),
    )


def _browse_request(controller, browse, label):
    return _foreign_request(
        browse, label, controller.selection.display_generation
    )


def _hold_browse_reads(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = _transport_api().read_frame_preview

    def held(*args, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", held)
    return entered, release


def test_e6pm2_same_token_rehydration_does_not_consume_a_stale_completion(
    monkeypatch, tmp_path
):
    """The ratified ABA row: equal token, different admission."""

    controller, _acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)
    request = _browse_request(controller, browse, 2)

    _demote_browse_publication(browse, 2)
    assert owner.submit(request) == request.token
    _settle_transport(transport)
    assert owner.consume_repaint() is True

    entered, release = _hold_browse_reads(monkeypatch)
    assert owner.submit(request) == request.token
    assert entered.wait(timeout=10.0)

    # Only the FIRST admission's completion is available; it must not satisfy
    # this one.
    assert owner.consume_repaint() is False
    assert owner.polling_needed() is True

    release.set()
    _settle_transport(transport)
    assert owner.consume_repaint() is True
    assert owner.polling_needed() is False


def test_e6pm2_rotated_out_completion_is_still_observed_exactly_once(
    monkeypatch, tmp_path
):
    controller, _acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)

    assert owner.submit(_browse_request(controller, browse, 2)) is not None
    _settle_transport(transport)

    # Push the owned completion out of the bounded 32-entry diagnostic deque
    # with FOREIGN traffic the owner does not track, BEFORE it ever looks, so
    # the rotated-out receipt is the only thing that can produce a wake.
    entered, release = _hold_browse_reads(monkeypatch)
    transport.submit(_foreign_request(browse, 1, 500))
    assert entered.wait(timeout=10.0)
    for generation in range(100, 134):
        transport.submit(_foreign_request(browse, 3, generation))
    assert len(transport.completions()) == 32
    owned = _admitted_browse_token(controller, browse, 2)
    assert all(
        completion.token != owned for completion in transport.completions()
    )

    release.set()
    _settle_transport(transport)
    # The consumer holds its own receipt, so rotation cannot lose the fact.
    assert owner.consume_repaint() is True
    assert owner.consume_repaint() is False


def test_e6pm2_active_and_queued_admissions_both_retain_terminal_facts(
    monkeypatch, tmp_path
):
    controller, _acquisition, browse, _processed = (
        _warm_browse_without_detector(tmp_path)
    )
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)

    entered, release = _hold_browse_reads(monkeypatch)
    first = _browse_request(controller, browse, 2)
    assert owner.submit(first) is not None
    assert entered.wait(timeout=10.0)
    # Same admission resubmitted: deduplicated by ticket identity.  (Done
    # BEFORE queuing B: a same-read resubmit legitimately supersedes a queued
    # entry, which is the accepted latest-selection-wins rule.)
    assert owner.submit(first) is not None
    # Active A plus queued B is normal; one slot would drop A's fact.
    assert owner.submit(_browse_request(controller, browse, 3)) is not None
    assert owner.polling_needed() is True

    release.set()
    _settle_transport(transport)
    assert owner.consume_repaint() is True
    assert owner.consume_repaint() is False
    assert owner.polling_needed() is False
    # BOTH admissions' terminal facts were applied, not just the latest.
    assert owner.detector_outcome(browse, 2) is not None
    assert owner.detector_outcome(browse, 3) is not None


def test_e6pm2_superseded_is_inert_on_borrowed_and_owned_paths(
    monkeypatch, tmp_path
):
    controller, _acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)

    entered, release = _hold_browse_reads(monkeypatch)
    assert owner.submit(_browse_request(controller, browse, 2)) is not None
    assert entered.wait(timeout=10.0)
    # Move the presentation on the SAME read: the displaced admission settles
    # SUPERSEDED and must neither terminalize nor repaint.
    moved = _foreign_request(
        browse, 2, controller.selection.display_generation + 1
    )
    assert owner.submit(moved) is not None
    assert transport.counters()[HydrationOutcome.SUPERSEDED] >= 1
    assert owner.consume_repaint() is False
    assert owner.detector_outcome(browse, 2) is None

    release.set()
    _settle_transport(transport)
    assert owner.consume_repaint() is True

    # The cold OWNED sink obeys the same rule.
    cold_controller, cold_browse, _p = _adopted_cold_browse(tmp_path / "cold")
    cold_owner = _bound_owner(cold_controller)
    cold_owner._complete(
        HydrationCompletion(
            _admitted_browse_token(cold_controller, cold_browse, 1),
            HydrationOutcome.SUPERSEDED,
        )
    )
    assert cold_owner.consume_repaint() is False
    assert cold_owner.detector_outcome(cold_browse, 1) is None


def test_e6pm2_borrowed_release_never_cancels_or_retires_and_keeps_tickets(
    monkeypatch, tmp_path
):
    controller, acquisition, browse, _processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)
    calls: list[str] = []
    monkeypatch.setattr(
        transport, "cancel_gate", lambda gate: calls.append("cancel_gate")
    )
    monkeypatch.setattr(
        transport, "retire", lambda **kwargs: calls.append("retire") or True
    )

    entered, release = _hold_browse_reads(monkeypatch)
    assert owner.submit(_browse_request(controller, browse, 2)) is not None
    assert entered.wait(timeout=10.0)

    receipt = controller.close()
    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    # Borrowed release touches neither lifecycle seam and keeps its receipts.
    assert calls == []
    assert owner.polling_needed() is True

    release.set()
    _settle_transport(transport)
    deadline = time.monotonic() + 10.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        receipt = controller.close()
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert calls == []
    assert transport is acquisition.publication_store.transport


def test_e6pm2_a_to_b_to_a_split_resolves_by_admission_identity(
    monkeypatch, tmp_path
):
    """Token values compare equal across A-to-B-to-A; admissions do not."""

    controller, _acquisition, browse, _processed = (
        _warm_browse_without_detector(tmp_path)
    )
    owner = _bound_owner(controller)
    transport = owner.transport
    _browse_key(controller, 2)
    generation = controller.selection.display_generation
    entered = threading.Event()
    release = threading.Event()
    original_commit = transport._commit

    def held_commit(prepared):
        entered.set()
        release.wait(timeout=10.0)
        return original_commit(prepared)

    monkeypatch.setattr(transport, "_commit", held_commit)

    first = transport._submit_admission(_foreign_request(browse, 2, generation))
    assert entered.wait(timeout=10.0)  # committed_token/ticket now snapshotted
    moved = transport._submit_admission(
        _foreign_request(browse, 2, generation + 1)
    )
    back = transport._submit_admission(
        _foreign_request(browse, 2, generation)
    )
    assert back is not first and back is not moved
    assert back.token == first.token  # equal VALUE, different admission

    release.set()
    _settle_transport(transport)
    # First-wins keeps the displaced facts; only the CURRENT admission takes
    # the already-resident terminal fact.
    assert first.result().outcome is HydrationOutcome.SUPERSEDED
    assert moved.result().outcome is HydrationOutcome.SUPERSEDED
    assert back.result().outcome is HydrationOutcome.ALREADY_RESIDENT


# E6-PM2 correction 1 (2026-08-04): a pending B-to-C retry must not discard B's
# settled borrowed receipt.  Replacement release preserves the wake; terminal
# close/retirement must never wait for a repaint their timer will not consume.


def _held_b_with_settled_ticket(monkeypatch, tmp_path):
    """Warm B whose borrowed read settles while its first release is pending."""

    controller, _acq, browse, processed = _adopted_browse(tmp_path)
    owner = _bound_owner(controller)
    _browse_key(controller, 2)
    _demote_browse_publication(browse, 2)
    entered, release = _hold_browse_reads(monkeypatch)
    assert owner.submit(_browse_request(controller, browse, 2)) is not None
    assert entered.wait(timeout=10.0)

    import pytest

    with pytest.raises(RuntimeError, match="previous Browse cleanup is pending"):
        controller.begin_browse(str(processed))
    assert controller.browse_context is browse
    deadline = time.monotonic() + 10.0
    while controller.browse_pending and time.monotonic() < deadline:
        controller.poll_browse()
        time.sleep(0.005)
    assert controller.browse_pending is False

    release.set()
    _settle_transport(owner.transport)
    assert owner.polling_needed() is True
    return controller, owner, browse, processed


def test_e6pm2_pending_b_to_c_retry_preserves_the_settled_b_wake(
    monkeypatch, tmp_path
):
    controller, owner, browse, processed = _held_b_with_settled_ticket(
        monkeypatch, tmp_path
    )

    import pytest

    # Retry before any timer tick: cleanup applies B's settled receipt first,
    # then holds B and its owner so the page poll can still reach the wake.
    with pytest.raises(RuntimeError, match="previous Browse cleanup is pending"):
        controller.begin_browse(str(processed))
    assert controller.browse_context is browse
    assert controller._browse_hydration_owner is owner

    # The normal controller repaint poll consumes exactly one B wake.
    assert controller.poll_browse_preview() is True
    assert controller.poll_browse_preview() is False

    # Only the NEXT exact replacement release may clean B.
    deadline = time.monotonic() + 10.0
    while controller.browse_pending and time.monotonic() < deadline:
        controller.poll_browse()
        time.sleep(0.005)
    controller.begin_browse(str(processed))
    assert controller.browse_context is not browse


def test_e6pm2_terminal_close_never_waits_for_a_pending_repaint(
    monkeypatch, tmp_path
):
    controller, owner, browse, _processed = _held_b_with_settled_ticket(
        monkeypatch, tmp_path
    )

    # close() is terminal/discarding: its presentation timer is stopped, so it
    # must finish rather than hold B for a wake nobody will consume.
    receipt = controller.close()
    deadline = time.monotonic() + 10.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        receipt = controller.close()
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert browse.released is True


def test_e6pm2_forged_cleaned_receipt_cannot_clear_b_owner_or_wake(
    monkeypatch, tmp_path
):
    from xdart.gui.tabs.scattering.browse_values import BrowseCleanupReceipt

    controller, owner, browse, _processed = _held_b_with_settled_ticket(
        monkeypatch, tmp_path
    )
    foreign = BrowseCleanupReceipt(None, CleanupStatus.CLEANED)
    monkeypatch.setattr(
        controller._browse_loader,
        "release_context",
        lambda _context: foreign,
    )

    receipt = controller._release_browse(browse)
    assert receipt is foreign
    # A mismatched request must not clear the bound owner, current B, or the
    # already-applied wake.
    assert controller._browse_hydration_owner is owner
    assert controller.browse_context is browse
    assert owner.consume_repaint() is True


def test_e6pm2_settled_ticket_display_retirement_releases_b_and_owner(
    monkeypatch, tmp_path
):
    from xdart.gui.tabs.scattering.display_retirement import (
        DisplayRetirementReceipt,
    )

    controller, owner, browse, _processed = _held_b_with_settled_ticket(
        monkeypatch, tmp_path
    )
    assert owner.polling_needed() is True

    # Display retirement is terminal/discarding like close(): it applies B's
    # settled receipt on the discarding path and must not hold B for a repaint
    # its timer will never consume.
    receipt = DisplayRetirementReceipt(
        controller.run_identity, CleanupStatus.CLEANED
    )
    assert controller.apply_display_retirement(receipt) is True
    assert browse.released is True
    assert controller.browse_context is None
    assert controller._browse_hydration_owner is None
    assert controller.run_identity is None
    assert controller.acquisition_context is None
