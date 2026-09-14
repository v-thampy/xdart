"""Frozen E6 Browse B oracle: typed one-read resolution and pass reuse.

Browse B replaces the issuer's second current-label store read and its
request construction/recomputation with ONE `ContextProjection.resolve_browse`
read producing a typed `BrowseProjectionResolution`, carried by a private
exact-pass snapshot that residency reuses without rereading current.

Every behavioral row here is frozen RED on exact parent ``abff4b77`` through
its own intended assertion (an explicit in-test failure when a Browse B
symbol is absent — never one shared collection-time import error).  The
architecture rows assert the issuer/owner-graph facts directly against the
production sources, so the surviving issuer second read is red at the parent
while Browse A's store-attachment sentinel stays green unchanged.

The §22.4 exact-scope qualification rows at the end were frozen RED on the
preserved pre-rework candidate (parent + tracked diff ``dac2a0a3``), each
through its intended assertion, before the §22.3 structural correction.
"""

from __future__ import annotations

import dataclasses
import inspect
import threading
import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from tests.xdart.scattering.test_e4_preview_transport import (
    _adopted_cold_browse,
    _browse_key,
    _demote_browse_publication,
    _transport_api,
)

from tests.xdart.scattering.test_e6_browse_hydration_owner import (
    _warm_browse_preview,
)

_VALUE_NAMES = (
    "BrowseMissReason",
    "QualifiedPayload",
    "HydrationEligibleMiss",
    "TerminalMiss",
    "BrowseProjectionResolution",
    "_BrowseProjectionPass",
)


def _values():
    """Import the Browse B value contract, red-by-assertion at the parent."""
    module = import_module("xdart.gui.tabs.scattering.context_values")
    missing = [name for name in _VALUE_NAMES if not hasattr(module, name)]
    if missing:
        pytest.fail(f"Browse B values absent at exact parent: {missing}")
    return module


def _require_resolver():
    from xdart.gui.tabs.scattering.context_projection import ContextProjection

    if getattr(ContextProjection, "resolve_browse", None) is None:
        pytest.fail("ContextProjection.resolve_browse absent at exact parent")
    return ContextProjection


def _production_sources() -> dict[str, str]:
    package = import_module("xdart.gui.tabs.scattering")
    root = Path(inspect.getfile(package)).parent
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted(root.rglob("*.py"))
    }


def _preview_source() -> str:
    return inspect.getsource(
        import_module("xdart.gui.tabs.scattering.browse_preview")
    )


def _spy_gets(monkeypatch, browse):
    """Count MAIN-THREAD store gets only: worker-side commit reads are the
    transport's own and never part of the projection sequence budget."""
    store = browse.publication_store
    real = store.get
    calls: list[object] = []

    def spying(label):
        if threading.current_thread() is threading.main_thread():
            calls.append(label)
        return real(label)

    monkeypatch.setattr(store, "get", spying)
    return calls


def _spy_submits(monkeypatch, controller):
    owner = controller._browse_hydration_owner
    transport = owner.transport
    real = transport.submit
    calls: list[tuple[Any, dict]] = []

    def spying(request, **kwargs):
        calls.append((request, kwargs))
        return real(request, **kwargs)

    monkeypatch.setattr(transport, "submit", spying)
    return calls


def _browse_pass(controller):
    return getattr(controller._runtime, "_browse_pass", None)


def _wait_repaint(controller, deadline_s: float = 10.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if controller.poll_browse_preview():
            return True
        time.sleep(0.005)
    return False


def _completed_cold_browse(tmp_path, label):
    """A sparse cold Browse explicitly hydrated before projection spies."""
    controller, browse, processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _warm_browse_preview(controller, label)
    return controller, browse, processed, key


# --------------------------------------------------------------------------- #
# Parent-polarity architecture rows (kickoff item 1 / census item 3)
# --------------------------------------------------------------------------- #

def test_issuer_performs_no_current_publication_store_read():
    assert "publication_store" not in _preview_source()


def test_issuer_constructs_no_hydration_request():
    assert "HydrationRequest" not in _preview_source()


def test_issuer_recomputes_no_hydration_eligibility():
    assert "publication_needs_hydration" not in _preview_source()


def test_no_production_reference_to_request_browse_preview():
    offending = [
        name
        for name, source in _production_sources().items()
        if "request_browse_preview" in source
    ]
    assert offending == []


def test_publication_needs_hydration_only_in_resolver_family():
    permitted = {
        "context_projection.py",
        "display_runtime.py",
        "browse_hydration.py",
    }
    referencing = {
        name
        for name, source in _production_sources().items()
        if "publication_needs_hydration" in source
    }
    assert referencing <= permitted, sorted(referencing - permitted)


def test_acquisition_closed_passthrough_reaches_detached_transport():
    source = inspect.getsource(
        import_module("xdart.gui.tabs.scattering.display_runtime")
    )
    assert "self._transport.submit_detached(request, closed=closed)" in source


# --------------------------------------------------------------------------- #
# Frozen value contract
# --------------------------------------------------------------------------- #

def test_resolution_values_frozen_slotted_without_resident_boolean():
    values = _values()
    reasons = {member.name for member in values.BrowseMissReason}
    assert reasons == {"CLOSED", "RETIRED", "FOREIGN", "UNRESOLVABLE"}

    expected_fields = {
        values.QualifiedPayload: ("payload",),
        values.HydrationEligibleMiss: ("request",),
        values.TerminalMiss: ("reason",),
        values._BrowseProjectionPass: (
            "selection",
            "current",
            "generation",
            "resolution",
        ),
    }
    for cls, names in expected_fields.items():
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen
        assert "__slots__" in cls.__dict__ or hasattr(cls, "__slots__")
        fields = dataclasses.fields(cls)
        assert tuple(field.name for field in fields) == names
        for field in fields:
            assert field.type is not bool and field.type != "bool"
            assert "resident" not in field.name.lower()

    with pytest.raises(TypeError):
        values.QualifiedPayload(object())
    with pytest.raises(TypeError):
        values.HydrationEligibleMiss(object())
    with pytest.raises(TypeError):
        values.TerminalMiss(object())


# --------------------------------------------------------------------------- #
# One-read resolution and pass reuse
# --------------------------------------------------------------------------- #

def test_complete_current_single_get_zero_submit_resident(
    monkeypatch, tmp_path
):
    _require_resolver()
    values = _values()
    controller, browse, _processed, key = _completed_cold_browse(tmp_path, 2)

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    payloads = controller.project_navigation()
    resident = controller.resident_frame_keys

    assert gets.count(2) == 1
    assert submits == []
    assert key in resident
    snapshot = _browse_pass(controller)
    assert snapshot is not None
    assert type(snapshot.resolution) is values.QualifiedPayload
    assert any(payload.frame_key is key for payload in payloads)
    assert snapshot.resolution.payload.view.thumbnail is not None


def test_incomplete_current_single_get_exact_submit_not_resident(
    monkeypatch, tmp_path
):
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _browse_key(controller, 2)
    _warm_browse_preview(controller, 2)
    _demote_browse_publication(browse, 2)

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    payloads = controller.project_navigation()
    resident = controller.resident_frame_keys

    assert gets.count(2) == 1
    assert len(submits) == 1
    request, kwargs = submits[0]
    assert not kwargs.get("closed", False)
    assert key not in resident
    assert not any(payload.frame_key is key for payload in payloads)
    snapshot = _browse_pass(controller)
    assert snapshot is not None
    assert type(snapshot.resolution) is values.HydrationEligibleMiss
    assert snapshot.resolution.request is request


def test_submitted_request_carries_exact_context_store_gate_artifact(
    monkeypatch, tmp_path
):
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    _browse_key(controller, 2)
    _warm_browse_preview(controller, 2)
    _demote_browse_publication(browse, 2)

    submits = _spy_submits(monkeypatch, controller)
    controller.project_navigation()

    assert len(submits) == 1
    request, _kwargs = submits[0]
    assert request.owner.as_tuple() == browse.hydration_owner.as_tuple()
    assert request.stores == (browse.publication_store,)
    assert request.commit_gate is browse.commit_gate
    assert request.read_key.artifact_identity == browse.requested_path
    assert request.read_key.frame_identity == 2
    assert request.generation == controller.selection.display_generation


def test_terminal_cold_completion_payload_resident_no_reenqueue(
    monkeypatch, tmp_path
):
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, thumbnails=False
    )
    key = _browse_key(controller, 2)
    controller.project_navigation()
    assert _wait_repaint(controller)

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    payloads = controller.project_navigation()
    resident = controller.resident_frame_keys

    assert gets.count(2) == 1
    assert submits == []
    assert key in resident
    snapshot = _browse_pass(controller)
    assert snapshot is not None
    assert type(snapshot.resolution) is values.QualifiedPayload
    payload = snapshot.resolution.payload
    assert payload.view.intensity_1d is not None
    assert payload.view.raw is None and payload.view.thumbnail is None
    assert any(candidate is payload for candidate in payloads)


def test_equal_valued_foreign_context_terminal_zero_reads_zero_submits(
    monkeypatch, tmp_path
):
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    key = _browse_key(controller, 2)
    foreign = replace(browse)
    assert foreign is not browse

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    controller._runtime._browse = foreign
    try:
        controller.project_navigation()
        snapshot = _browse_pass(controller)
        assert snapshot is not None
        assert type(snapshot.resolution) is values.TerminalMiss
        assert snapshot.resolution.reason is values.BrowseMissReason.FOREIGN
        assert gets == []
        assert submits == []
        # Residency may inspect OTHER frames; the current label is neither
        # resident nor reread through the foreign context.
        assert key not in controller.resident_frame_keys
        assert gets.count(2) == 0
    finally:
        controller._runtime._browse = browse


def test_retired_context_terminal_zero_reads(monkeypatch, tmp_path):
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    _browse_key(controller, 2)
    browse.invalidate()

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    controller.project_navigation()

    snapshot = _browse_pass(controller)
    if snapshot is not None:
        assert type(snapshot.resolution) is values.TerminalMiss
        assert snapshot.resolution.reason is values.BrowseMissReason.RETIRED
    assert gets == []
    assert submits == []


def test_closed_gate_terminal_zero_reads(monkeypatch, tmp_path):
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    _browse_key(controller, 2)
    browse.commit_gate.cancel()

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    controller.project_navigation()

    snapshot = _browse_pass(controller)
    assert snapshot is not None
    assert type(snapshot.resolution) is values.TerminalMiss
    assert snapshot.resolution.reason is values.BrowseMissReason.CLOSED
    assert gets == []
    assert submits == []


def test_repeated_label_change_reads_only_new_current(monkeypatch, tmp_path):
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(tmp_path)
    _browse_key(controller, 2)
    controller.project_navigation()

    _warm_browse_preview(controller, 3)
    _demote_browse_publication(browse, 3)
    key3 = _browse_key(controller, 3)
    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    controller.project_navigation()

    assert gets.count(3) == 1
    assert gets.count(2) == 0
    snapshot = _browse_pass(controller)
    assert snapshot is not None and snapshot.current is key3
    assert len(submits) == 1
    request, _kwargs = submits[0]
    assert request.owner.as_tuple() == browse.hydration_owner.as_tuple()
    assert request.stores == (browse.publication_store,)
    assert request.commit_gate is browse.commit_gate
    assert request.read_key.artifact_identity == browse.requested_path
    assert request.read_key.frame_identity == 3


def test_navigation_change_invalidates_pass_without_stale_reuse_or_reread(
    monkeypatch, tmp_path
):
    _require_resolver()
    controller, browse, _processed, _key = _completed_cold_browse(tmp_path, 2)
    controller.project_navigation()
    assert _browse_pass(controller) is not None

    key3 = _browse_key(controller, 3)
    assert _browse_pass(controller) is None

    gets = _spy_gets(monkeypatch, browse)
    resident = controller.resident_frame_keys
    assert key3 not in resident
    assert gets.count(3) == 0


def test_selection_generation_change_invalidates_pass(tmp_path):
    _require_resolver()
    controller, _browse, _processed, _key = _completed_cold_browse(tmp_path, 2)
    controller.project_navigation()
    assert _browse_pass(controller) is not None

    before = controller.selection.display_generation
    controller.select_browse()
    assert controller.selection.display_generation > before
    assert _browse_pass(controller) is None


def test_resolver_exception_and_malformed_result_leave_no_pass(
    monkeypatch, tmp_path
):
    _require_resolver()
    controller, _browse, _processed, key = _completed_cold_browse(tmp_path, 2)

    def raising(*_args, **_kwargs):
        raise RuntimeError("resolver failure")

    monkeypatch.setattr(controller._projection, "resolve_browse", raising)
    payloads = controller.project_navigation()
    assert _browse_pass(controller) is None
    assert not any(payload.frame_key is key for payload in payloads)
    assert key not in controller.resident_frame_keys

    monkeypatch.setattr(
        controller._projection,
        "resolve_browse",
        lambda *_args, **_kwargs: object(),
    )
    payloads = controller.project_navigation()
    assert _browse_pass(controller) is None
    assert not any(payload.frame_key is key for payload in payloads)


def test_refused_submit_leaves_no_pass_fail_closed(monkeypatch, tmp_path):
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _browse_key(controller, 2)
    _warm_browse_preview(controller, 2)
    _demote_browse_publication(browse, 2)

    owner = controller._browse_hydration_owner
    monkeypatch.setattr(
        owner.transport, "submit", lambda _request, **_kwargs: None
    )
    controller.project_navigation()

    assert _browse_pass(controller) is None
    assert key not in controller.resident_frame_keys


def test_stale_refusal_regenerates_equivalent_current_miss(
    monkeypatch, tmp_path
):
    """Kickoff row: a refused stale miss must never strand cold hydration."""
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _browse_key(controller, 2)
    _warm_browse_preview(controller, 2)
    _demote_browse_publication(browse, 2)

    entered = threading.Event()
    release = threading.Event()
    preview_module = import_module("xrd_tools.io.frame_preview")
    original = preview_module.read_frame_preview

    def holding_read(read_key, **kwargs):
        entered.set()
        release.wait(timeout=10.0)
        return original(read_key, **kwargs)

    monkeypatch.setattr(_transport_api(), "read_frame_preview", holding_read)
    try:
        submits = _spy_submits(monkeypatch, controller)
        controller.project_navigation()
        first = _browse_pass(controller)
        assert first is not None
        assert type(first.resolution) is values.HydrationEligibleMiss
        miss_n = first.resolution.request
        generation_n = first.generation
        assert [request for request, _ in submits] == [miss_n]

        controller.select_browse()  # exact-pass staleness at N+1
        assert _browse_pass(controller) is None

        controller.project_navigation()
        second = _browse_pass(controller)
        assert second is not None
        assert type(second.resolution) is values.HydrationEligibleMiss
        miss_next = second.resolution.request
        assert miss_next is not miss_n
        assert second.generation == generation_n + 1
        assert miss_next.generation == generation_n + 1
        assert miss_next.read_key.frame_identity == 2
        assert miss_next.read_key.artifact_identity == browse.requested_path
        assert miss_next.owner.as_tuple() == miss_n.owner.as_tuple()
        assert miss_next.stores == (browse.publication_store,)
        assert miss_next.commit_gate is browse.commit_gate
        submitted = [request for request, _ in submits]
        assert submitted == [miss_n, miss_next]
        assert submitted[1] is miss_next
        assert key not in controller.resident_frame_keys
    finally:
        release.set()


def test_one_total_current_get_across_projection_then_residency(
    monkeypatch, tmp_path
):
    _require_resolver()
    controller, browse, _processed, key = _completed_cold_browse(tmp_path, 2)

    gets = _spy_gets(monkeypatch, browse)
    controller.project_navigation()
    resident = controller.resident_frame_keys
    assert key in resident
    assert gets.count(2) == 1


# --------------------------------------------------------------------------- #
# §22.4 authoritative exact-scope qualification rows
# --------------------------------------------------------------------------- #

def test_stale_resolution_is_refused_before_submit(monkeypatch, tmp_path):
    """§22.4 row 1: view drift after resolver return refuses the stale miss
    before submit; the next stable pass regenerates and admits a fresh
    current-generation miss."""
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    _browse_key(controller, 2)
    _warm_browse_preview(controller, 2)
    _demote_browse_publication(browse, 2)

    projection = controller._projection
    resolve = projection.resolve_browse

    def resolve_then_advance(*args, **kwargs):
        result = resolve(*args, **kwargs)
        controller.select_browse()
        return result

    submits = _spy_submits(monkeypatch, controller)
    monkeypatch.setattr(projection, "resolve_browse", resolve_then_advance)
    controller.project_navigation()

    assert submits == []
    assert _browse_pass(controller) is None

    monkeypatch.setattr(projection, "resolve_browse", resolve)
    controller.project_navigation()
    assert len(submits) == 1
    request, _kwargs = submits[0]
    assert request.generation == controller.selection.display_generation


def test_typed_but_wrong_payload_is_malformed_and_not_resident(
    monkeypatch, tmp_path
):
    """§22.4 row 2: typed wrong-generation and wrong-frame qualified payloads
    are malformed at the install boundary: no pass, projection or residency."""
    _require_resolver()
    values = _values()
    controller, _browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _warm_browse_preview(controller, 2)
    payload = next(
        item for item in controller.project_navigation()
        if item.frame_key is key
    )
    other = next(frame for frame in controller.frame_keys if frame is not key)
    for wrong in (
        replace(payload, selection_generation=payload.selection_generation + 1),
        replace(payload, frame_key=other),
    ):
        controller._runtime.invalidate_browse_pass()
        malformed = values.QualifiedPayload(wrong)
        monkeypatch.setattr(
            controller._projection,
            "resolve_browse",
            lambda *_args, _m=malformed, **_kwargs: _m,
        )
        assert controller.project_navigation() == ()
        assert _browse_pass(controller) is None
        assert key not in controller.resident_frame_keys


def test_retained_pass_fails_closed_after_context_invalidation(tmp_path):
    """§22.4 row 3: direct Browse invalidation disqualifies the retained pass
    on direct reuse, with no prior navigation call."""
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _warm_browse_preview(controller, 2)
    assert any(
        item.frame_key is key for item in controller.project_navigation()
    )
    assert _browse_pass(controller) is not None

    browse.invalidate()

    assert controller.project(key) is None
    assert key not in controller.resident_frame_keys
    assert _browse_pass(controller) is None


def test_retained_pass_fails_closed_after_context_release(
    monkeypatch, tmp_path
):
    """§22.4 row 3 release variant, residency first: the disqualified pass is
    cleared by residency itself, never counted resident, never reread."""
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _warm_browse_preview(controller, 2)
    controller.project_navigation()
    assert _browse_pass(controller) is not None

    receipt = controller._browse_hydration_owner.release(
        controller._browse_loader, browse, preserve_pending_repaint=False,
    )
    assert receipt.cleanup_status.value == "cleaned"
    assert browse.released

    gets = _spy_gets(monkeypatch, browse)
    assert key not in controller.resident_frame_keys
    assert _browse_pass(controller) is None
    assert controller.project(key) is None
    assert gets.count(2) == 0


def test_retained_pass_fails_closed_after_direct_gate_cancel(
    monkeypatch, tmp_path
):
    """§22.4 row 3 gate-cancel variant: navigation, residency and direct reuse
    all fail closed with zero rereads and zero submits."""
    _require_resolver()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _warm_browse_preview(controller, 2)
    controller.project_navigation()
    assert _browse_pass(controller) is not None

    browse.commit_gate.cancel()

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    assert controller.project(key) is None
    assert key not in controller.resident_frame_keys
    assert not any(
        item.frame_key is key for item in controller.project_navigation()
    )
    assert gets.count(2) == 0
    assert submits == []


def test_equal_valued_foreign_selection_is_terminal_before_read(
    monkeypatch, tmp_path
):
    """§22.4 row 4 (architecture/test seam per §22.10 note 2): the runtime-
    owned anchors reject an equal-valued non-identical selection or frame by
    identity before the one store read, with zero reads and zero submits."""
    _require_resolver()
    values = _values()
    controller, browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _browse_key(controller, 2)
    runtime = controller._runtime
    request = runtime.project_request(key)
    clone_selection = replace(request, selection=replace(request.selection))
    assert clone_selection.selection is not request.selection
    assert clone_selection.selection == request.selection
    clone_frame = replace(request, frame=replace(key))
    assert clone_frame.frame is not key and clone_frame.frame == key

    gets = _spy_gets(monkeypatch, browse)
    submits = _spy_submits(monkeypatch, controller)
    for forged in (clone_selection, clone_frame):
        resolution = controller._projection.resolve_browse(
            browse,
            forged,
            runtime._selection,
            runtime._browse_navigation.current,
            controller._browse_hydration_owner,
        )
        assert type(resolution) is values.TerminalMiss
        assert resolution.reason is values.BrowseMissReason.FOREIGN
    assert gets == []
    assert submits == []


def test_pass_reuse_is_keyed_by_object_identity_not_equality(tmp_path):
    """A pass whose selection is an equal-valued clone of the live view is
    disqualified and cleared: reuse is keyed by exact object identity."""
    _require_resolver()
    values = _values()
    controller, _browse, _processed = _adopted_cold_browse(
        tmp_path, loader_max=32
    )
    key = _browse_key(controller, 2)
    controller.project_navigation()
    snapshot = _browse_pass(controller)
    assert snapshot is not None

    controller._runtime._browse_pass = values._BrowseProjectionPass(
        replace(snapshot.selection),
        snapshot.current,
        snapshot.generation,
        snapshot.resolution,
    )
    assert controller.project(key) is None
    assert _browse_pass(controller) is None
