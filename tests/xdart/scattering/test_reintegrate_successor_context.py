"""Focused immutable Reintegration successor Browse custody oracles."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.io.output_transaction import StreamTerminal

from tests.xdart.scattering.test_e3_context_contract import (
    _browse,
    _cold_controller,
    _select_browse,
)


def _terminal(target: str) -> StreamTerminal:
    return StreamTerminal(target, 17, "d" * 64, 1, 2, 3, 4, 5)


def test_ready_successor_swaps_before_retrying_retired_predecessor(
    monkeypatch,
) -> None:
    controller, _lifecycle, loader = _cold_controller()
    request_b, browse_b = _select_browse(
        controller, loader, scan_key="immutable.b",
    )
    capture_b = controller.capture_loaded_browse(request_b)
    assert capture_b is not None
    old_owner = controller._browse_hydration_owner
    owner = object()
    target_c = "/processed/immutable.c.nxs"
    terminal_c = _terminal(target_c)
    request_c = controller.begin_reintegrate_successor_browse(
        capture_b, target_c, terminal_c, owner=owner,
        expected_entry=capture_b.entry,
        expected_labels=capture_b.labels,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="immutable.c",
        request=request_c,
    )
    loader.complete(browse_c)

    owner_type = type(old_owner)
    real_release = owner_type.release
    calls = []

    def fail_old_once(self, bound_loader, browse, **kwargs):
        if self is old_owner and not calls:
            calls.append(browse)
            browse.invalidate()
            from xdart.gui.tabs.scattering.browse_values import (
                BrowseCleanupReceipt,
            )
            return BrowseCleanupReceipt(
                browse.load_request, CleanupStatus.CLEANUP_PENDING,
            )
        return real_release(self, bound_loader, browse, **kwargs)

    monkeypatch.setattr(owner_type, "release", fail_old_once)
    outcome = controller.poll_browse(reintegrate_successor_owner=owner)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    assert controller.browse_context is browse_c
    capture_c = controller.capture_loaded_browse(request_c)
    assert capture_c is not None and capture_c.target == target_c
    assert controller._retired_browse_cleanup.context is browse_b
    assert controller._retired_browse_cleanup.owner is old_owner
    assert controller.browse_pending
    assert browse_b.invalidated and not browse_b.released

    controller.poll_browse()
    assert controller._retired_browse_cleanup is None
    assert browse_b.released
    assert controller.browse_context is browse_c
    assert controller.capture_loaded_browse(request_c) is not None
    assert controller.close().cleanup_status is CleanupStatus.CLEANED


def test_foreign_successor_owner_cannot_replace_live_predecessor() -> None:
    controller, _lifecycle, loader = _cold_controller()
    request_b, browse_b = _select_browse(
        controller, loader, scan_key="immutable.owner.b",
    )
    capture_b = controller.capture_loaded_browse(request_b)
    assert capture_b is not None
    owner = object()
    target_c = "/processed/immutable.owner.c.nxs"
    terminal_c = _terminal(target_c)
    request_c = controller.begin_reintegrate_successor_browse(
        capture_b, target_c, terminal_c, owner=owner,
        expected_entry=capture_b.entry,
        expected_labels=capture_b.labels,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="immutable.owner.c",
        request=request_c,
    )
    loader.complete(browse_c)

    assert controller.poll_browse(
        reintegrate_successor_owner=object(),
    ) is None
    assert controller.browse_context is browse_b
    assert controller.capture_loaded_browse(request_b) is not None
    assert controller._retired_browse_cleanup is None
    assert request_c in loader.cancelled


def test_wrong_successor_entry_cannot_replace_live_predecessor() -> None:
    controller, _lifecycle, loader = _cold_controller()
    request_b, browse_b = _select_browse(
        controller, loader, scan_key="immutable.entry.b",
    )
    capture_b = controller.capture_loaded_browse(request_b)
    assert capture_b is not None
    owner = object()
    target_c = "/processed/immutable.entry.c.nxs"
    terminal_c = _terminal(target_c)
    request_c = controller.begin_reintegrate_successor_browse(
        capture_b, target_c, terminal_c, owner=owner,
        expected_entry=capture_b.entry,
        expected_labels=capture_b.labels,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="immutable.entry.c",
        request=request_c,
    )
    browse_c = replace(browse_c, target_entry="wrong-entry")
    loader.complete(browse_c)

    assert controller.poll_browse(
        reintegrate_successor_owner=owner,
    ) is None
    assert controller.browse_context is browse_b
    assert controller.capture_loaded_browse(request_b) is not None
    assert request_c in loader.cancelled
    assert controller.close().cleanup_status is CleanupStatus.CLEANED


def test_numeric_equal_foreign_successor_labels_cannot_replace_predecessor(
) -> None:
    controller, _lifecycle, loader = _cold_controller()
    request_b, browse_b = _select_browse(
        controller, loader, scan_key="immutable.numeric-label.b",
    )
    capture_b = controller.capture_loaded_browse(request_b)
    assert capture_b is not None
    owner = object()
    target_c = "/processed/immutable.numeric-label.c.nxs"
    request_c = controller.begin_reintegrate_successor_browse(
        capture_b,
        target_c,
        _terminal(target_c),
        owner=owner,
        expected_entry=capture_b.entry,
        expected_labels=capture_b.labels,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="immutable.numeric-label.c",
        request=request_c,
    )
    foreign_labels = (1.0,)
    object.__setattr__(browse_c, "frame_ids", foreign_labels)
    object.__setattr__(browse_c, "loaded_labels", foreign_labels)
    loader.complete(browse_c)

    assert controller.poll_browse(
        reintegrate_successor_owner=owner,
    ) is None
    assert controller.browse_context is browse_b
    assert controller.capture_loaded_browse(request_b) is not None
    assert request_c in loader.cancelled
    assert controller.close().cleanup_status is CleanupStatus.CLEANED


def test_text_equal_foreign_successor_path_cannot_replace_predecessor() -> None:
    class ForeignPath(str):
        pass

    controller, _lifecycle, loader = _cold_controller()
    request_b, browse_b = _select_browse(
        controller, loader, scan_key="immutable.foreign-path.b",
    )
    capture_b = controller.capture_loaded_browse(request_b)
    assert capture_b is not None
    owner = object()
    target_c = "/processed/immutable.foreign-path.c.nxs"
    request_c = controller.begin_reintegrate_successor_browse(
        capture_b,
        target_c,
        _terminal(target_c),
        owner=owner,
        expected_entry=capture_b.entry,
        expected_labels=capture_b.labels,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="immutable.foreign-path.c",
        request=request_c,
    )
    object.__setattr__(
        browse_c,
        "requested_path",
        ForeignPath(request_c.source_path),
    )
    loader.complete(browse_c)

    assert controller.poll_browse(
        reintegrate_successor_owner=owner,
    ) is None
    assert controller.browse_context is browse_b
    assert controller.capture_loaded_browse(request_b) is not None
    assert request_c in loader.cancelled
    assert controller.close().cleanup_status is CleanupStatus.CLEANED


def test_worker_result_gate_authenticates_successor_inventory() -> None:
    from xrd_tools.reduction import ReintegrateSuccessorResult
    from xrd_tools.reduction import reintegrate_successor as successor_core

    source = "/processed/source.nexus"
    output = "/processed/source-reintegrated.nexus"
    labels = (2, 5, 9)
    plan = SimpleNamespace(
        source_artifact=source,
        output_artifact=output,
        labels=labels,
        science_identity="a" * 64,
        version_identity="b" * 64,
        publication_identity="c" * 64,
        operation_identity="d" * 64,
    )
    terminal = _terminal(output)

    def result(
        *,
        disposition="COMMITTED",
        input_labels=labels,
        committed_labels=(2, 9),
        dropped_labels=(5,),
        audit_identity="e" * 64,
        commit_identity="f" * 64,
        result_terminal=terminal,
        hidden_orphan=None,
    ):
        return successor_core._value(
            ReintegrateSuccessorResult,
            disposition,
            source,
            output,
            input_labels,
            committed_labels,
            dropped_labels,
            (),
            plan.science_identity,
            plan.version_identity,
            plan.publication_identity,
            plan.operation_identity,
            audit_identity,
            commit_identity,
            result_terminal,
            hidden_orphan,
        )

    valid = OperationSlot._valid_reintegrate_successor_result
    assert valid(result(), plan, ReintegrateSuccessorResult)
    assert not valid(
        result(committed_labels=(2, 5), dropped_labels=(5, 9)),
        plan,
        ReintegrateSuccessorResult,
    )
    assert not valid(
        result(committed_labels=(2,), dropped_labels=(5,)),
        plan,
        ReintegrateSuccessorResult,
    )
    assert not valid(
        result(input_labels=(2.0, 5.0, 9.0)),
        plan,
        ReintegrateSuccessorResult,
    )
    assert not valid(
        result(audit_identity="not-a-sha256"),
        plan,
        ReintegrateSuccessorResult,
    )
    assert not valid(
        result(hidden_orphan="/processed/contradictory-orphan.nexus"),
        plan,
        ReintegrateSuccessorResult,
    )
    aborted = result(
        disposition="ABORTED",
        committed_labels=(),
        dropped_labels=(2,),
        audit_identity=None,
        commit_identity=None,
        result_terminal=None,
    )
    assert valid(aborted, plan, ReintegrateSuccessorResult)
    assert not valid(
        result(
            disposition="ABORTED",
            committed_labels=(),
            dropped_labels=(2,),
            audit_identity="bad",
            commit_identity=None,
            result_terminal=None,
        ),
        plan,
        ReintegrateSuccessorResult,
    )
    assert not valid(
        result(
            disposition="ABORTED",
            committed_labels=(),
            dropped_labels=(12,),
            audit_identity=None,
            commit_identity=None,
            result_terminal=None,
        ),
        plan,
        ReintegrateSuccessorResult,
    )
