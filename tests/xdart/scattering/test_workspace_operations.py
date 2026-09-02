"""Qt-free ownership oracle for Average and Reintegrate composition."""

from __future__ import annotations

from dataclasses import replace

import pytest

from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
    LoadedBrowseCapture,
)
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationProgress,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.workspace_operations import (
    AverageOperationState,
    WorkspaceOperationOwner,
    WorkspaceRefreshEffect,
)
from xdart.gui.tabs.scattering.processed_browser import (
    AverageReloadDirective,
    ReintegrateReloadDirective,
)
from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
    HydrationOwner,
)
from xrd_tools.io.output_transaction import StreamTerminal, TargetSnapshot
from xrd_tools.reduction import (
    AverageFiniteCountsEvidence,
    AverageScanResult,
    ReintegrateSuccessorResult,
)
from xrd_tools.reduction import reintegrate_successor as successor_core
from xrd_tools.session.run_configuration import RunIntent


def _seal(target: str, ordinal: int = 1) -> StreamTerminal:
    return StreamTerminal(
        target, 17, "d" * 64, ordinal, 2, 3, 4, 5
    )


def _capture(target: str = "/detached/scan.nexus") -> LoadedBrowseCapture:
    from xrd_tools.reduction import prepare_reintegrate_bundle

    request = BrowseLoadRequest(
        "capture-token", 1, target, source_root="/detached"
    )
    context = object.__new__(BrowseContext)
    offer = prepare_reintegrate_bundle(
        None, entry="entry", labels=(0, 1, 2),
    )
    object.__setattr__(context, "prepared_reintegrate_offer", offer)
    selection = DisplaySelection(
        ContextKind.BROWSE,
        HydrationOwner("capture-token", "scan", target, 1),
        2,
    )
    return LoadedBrowseCapture(
        context,
        request,
        selection,
        target,
        "entry",
        TargetSnapshot(True, 17, 4, 2, 3, "f" * 64),
        (0, 1, 2),
        offer,
    )


def _reintegrate_result(
    capture: LoadedBrowseCapture,
    *,
    disposition: str = "COMMITTED",
    seal: StreamTerminal | None = None,
    diagnostics: tuple[str, ...] = (),
    hidden_orphan: str | None = None,
) -> ReintegrateSuccessorResult:
    output = capture.target.replace(".nexus", "-vnext.nexus")
    if disposition in {"COMMITTED", "ALREADY_COMMITTED"} and seal is None:
        seal = _seal(output)
    committed = disposition in {"COMMITTED", "ALREADY_COMMITTED"}
    return successor_core._value(
        ReintegrateSuccessorResult,
        disposition,
        capture.target,
        output,
        capture.labels,
        capture.labels if committed else (),
        (),
        diagnostics,
        "b" * 64,
        "a" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64 if committed else None,
        "f" * 64 if committed else None,
        seal if committed else None,
        hidden_orphan,
    )


def _stamp(
    capture: LoadedBrowseCapture, revision: int = 1,
) -> OperationContextStamp:
    return OperationContextStamp(
        revision,
        capture.selection.context_token,
        capture.selection.display_generation,
    )


def _average_result(target: str) -> AverageScanResult:
    return AverageScanResult(
        disposition="COMMITTED",
        target=target,
        entry="entry",
        operation_identity="a" * 64,
        science_identity="b" * 64,
        contributor_extent=2,
        logical_labels=(1,),
        committed_labels=(1,),
        metadata_denominators=(("I0", 2),),
        finite_counts=AverageFiniteCountsEvidence(
            "average_scan_v1",
            2,
            (1, 1),
            "<u4",
            "c" * 64,
            1,
            2,
            0,
            (1, 1),
            "gzip",
            1,
            True,
            False,
        ),
        diagnostic_code="",
        diagnostic="",
        h23_phase="committed",
        commit_identity=_seal(target),
    )


class _Slot:
    def __init__(self) -> None:
        self._identity: OperationIdentity | None = None
        self.next_identity: OperationIdentity | None = None
        self.reintegrate_calls = []
        self.average_calls = []
        self.cancel_calls = []
        self.retry_calls = []
        self.observed = []
        self.close_receipts = []

    @property
    def owned(self) -> bool:
        return self._identity is not None

    @property
    def current_identity(self) -> OperationIdentity | None:
        return self._identity

    def begin_reintegrate_successor(self, **kwargs):
        self.reintegrate_calls.append(kwargs)
        self._identity = self.next_identity
        return self.next_identity

    def begin_average(self, configuration, target, **kwargs):
        self.average_calls.append((configuration, target, kwargs))
        self._identity = self.next_identity
        return self.next_identity

    def cancel(self, identity):
        self.cancel_calls.append(identity)
        return identity is self._identity

    def retry_average(self, identity, pending):
        self.retry_calls.append((identity, pending))
        return identity is self._identity

    def observe_stamp(self, stamp):
        self.observed.append(stamp)

    def poll(self, _identity):
        return None

    def close(self):
        return self.close_receipts.pop(0)


def _owner_with_slot() -> tuple[WorkspaceOperationOwner, _Slot]:
    owner = WorkspaceOperationOwner()
    slot = _Slot()
    owner._slot = slot
    return owner, slot


def test_typed_capture_requires_live_identity_but_compares_value_facts() -> None:
    capture = _capture()
    same_values = replace(capture)
    assert capture.is_exactly(same_values)
    foreign = replace(capture, request=replace(capture.request))
    assert not capture.is_exactly(foreign)
    assert "__bool__" not in WorkspaceRefreshEffect.__dict__


def test_reload_directives_require_the_seal_to_name_the_exact_target() -> None:
    capture = _capture()
    foreign = _seal("/detached/foreign.nexus")
    with pytest.raises(ValueError, match="Reintegrate reload directive"):
        ReintegrateReloadDirective(
            capture.request, capture.target, foreign
        )
    with pytest.raises(ValueError, match="Average reload directive"):
        AverageReloadDirective(capture.target, capture.entry, foreign)


def test_reintegrate_begin_transfers_exact_capture_without_reload_custody(
) -> None:
    owner, slot = _owner_with_slot()
    capture = _capture()
    identity = OperationIdentity(7)
    slot.next_identity = identity
    preparation = {"api_version": 1}
    assert owner.begin_reintegrate(
        capture,
        dimension="2d",
        preparation_values=preparation,
        stamp=_stamp(capture, 3),
    ) is identity
    assert owner.reintegrate_identity is identity
    assert owner.reintegrate_capture is capture
    call = slot.reintegrate_calls[-1]
    assert call["source_artifact"] == capture.target
    assert call["entry"] == capture.entry
    assert call["expected_target_snapshot"] is capture.target_snapshot
    assert call["expected_labels"] is capture.labels
    assert call["source_root"] == capture.request.source_root
    assert call["prepared_offer"] is capture.prepared_reintegrate_offer
    assert call["dimension"] == "2d"
    assert call["preparation_values"] is preparation

    refused, refused_slot = _owner_with_slot()
    assert refused.begin_reintegrate(
        capture,
        dimension="1d",
        preparation_values=preparation,
        stamp=_stamp(capture, 3),
    ) is None
    assert refused.reintegrate_state is None
    assert not refused.busy
    assert refused_slot.reintegrate_calls


def test_reintegrate_progress_transition_is_exact_monotonic_and_cancel_fenced(
) -> None:
    owner, slot = _owner_with_slot()
    capture = _capture()
    identity = OperationIdentity(9)
    slot.next_identity = identity
    assert owner.begin_reintegrate(
        capture,
        dimension="1d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is identity

    first = OperationProgress(identity, "integrate", 2, 5, 2)
    transition = owner.consume_reintegrate_update(
        OperationUpdate(identity, progress=first)
    )
    assert transition.effect is WorkspaceRefreshEffect.NONE
    assert transition.reintegrate_progress is first
    assert transition.notice == "Reintegrate 1-D: integrate 2/5…"
    assert owner.reintegrate_state.progress is first

    foreign_identity = OperationIdentity(identity.serial)
    foreign = OperationProgress(foreign_identity, "integrate", 3, 5, 3)
    reused = OperationProgress(identity, "write", 3, 5, 2)
    regressed = OperationProgress(identity, "integrate", 1, 5, 3)
    stale = OperationProgress(identity, "integrate", 3, 5, 4)
    for update in (
        OperationUpdate(foreign_identity, progress=foreign),
        OperationUpdate(identity, progress=reused),
        OperationUpdate(identity, progress=regressed),
        OperationUpdate(identity, progress=stale, stale=True),
    ):
        ignored = owner.consume_reintegrate_update(update)
        assert ignored.effect is WorkspaceRefreshEffect.NONE
        assert ignored.reintegrate_progress is None
        assert owner.reintegrate_state.progress is first

    latest = OperationProgress(identity, "write", 3, 5, 5)
    accepted = owner.consume_reintegrate_update(
        OperationUpdate(identity, progress=latest)
    )
    assert accepted.reintegrate_progress is latest
    assert owner.reintegrate_state.progress is latest

    assert not owner.cancel_reintegrate("2d")
    assert not owner.reintegrate_cancel_accepted
    assert owner.cancel_reintegrate("1d")
    assert owner.reintegrate_cancel_accepted
    assert slot.cancel_calls == [identity]
    assert not owner.cancel_reintegrate("1d")
    late = owner.consume_reintegrate_update(OperationUpdate(
        identity,
        progress=OperationProgress(identity, "write", 4, 5, 6),
    ))
    assert late.reintegrate_progress is None
    assert owner.reintegrate_state.progress is latest


def test_reintegrate_terminal_adopts_successor_and_lost_owner_keeps_predecessor() -> None:
    capture = _capture()
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(11)
    slot.next_identity = identity
    assert owner.begin_reintegrate(
        capture,
        dimension="1d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is identity
    successor = capture.target.replace(".nexus", "-vnext.nexus")
    seal = _seal(successor)
    result = _reintegrate_result(capture, seal=seal)
    transition = owner.consume_reintegrate_update(OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result
        ),
    ))
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert owner.reintegrate_state is None
    assert transition.reintegrate_reload is None
    assert transition.reintegrate_successor is not None
    assert transition.reintegrate_successor.predecessor is capture
    assert transition.reintegrate_successor.successor_path == successor
    assert transition.reintegrate_successor.terminal_commit_identity is seal
    assert owner.reintegrate_state is None

    lost, lost_slot = _owner_with_slot()
    lost_identity = OperationIdentity(12)
    lost_slot.next_identity = lost_identity
    assert lost.begin_reintegrate(
        capture,
        dimension="2d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is lost_identity
    transition = lost.consume_lost_owner(lost_identity)
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert "before terminal publication" in transition.notice
    assert lost.reintegrate_state is None
    assert transition.reintegrate_reload is None
    assert transition.reintegrate_successor is None


def test_reintegrate_stop_race_keeps_current_owner_but_abandonment_does_not() -> None:
    capture = _capture()

    stopped, stopped_slot = _owner_with_slot()
    stopped_identity = OperationIdentity(13)
    stopped_slot.next_identity = stopped_identity
    assert stopped.begin_reintegrate(
        capture,
        dimension="1d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is stopped_identity
    assert stopped.cancel_reintegrate("1d")
    committed = _reintegrate_result(capture)
    transition = stopped.consume_reintegrate_update(OperationUpdate(
        stopped_identity,
        terminal=OperationTerminal(
            stopped_identity,
            OperationTerminalStatus.RETURNED,
            payload=committed,
        ),
    ))
    assert transition.reintegrate_successor is not None
    assert transition.request_catalog

    abandoned, abandoned_slot = _owner_with_slot()
    abandoned_identity = OperationIdentity(14)
    abandoned_slot.next_identity = abandoned_identity
    assert abandoned.begin_reintegrate(
        capture,
        dimension="2d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is abandoned_identity
    assert abandoned.abandon_reintegrate(abandoned_identity)
    assert abandoned.reintegrate_state.owner_abandoned
    transition = abandoned.consume_reintegrate_update(OperationUpdate(
        abandoned_identity,
        terminal=OperationTerminal(
            abandoned_identity,
            OperationTerminalStatus.RETURNED,
            payload=committed,
        ),
    ))
    assert transition.reintegrate_successor is None
    assert transition.request_catalog
    assert "display owner changed" in transition.notice


def test_reintegrate_surfaces_fallback_and_hidden_orphan_diagnostics() -> None:
    capture = _capture()
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(15)
    slot.next_identity = identity
    assert owner.begin_reintegrate(
        capture,
        dimension="1d",
        preparation_values={"api_version": 1},
        stamp=_stamp(capture),
    ) is identity
    hidden = "/detached/.xdart-orphan"
    aborted = _reintegrate_result(
        capture,
        disposition="ABORTED",
        diagnostics=("PREPARED_CAPSULE_MISS:SOURCE_TOPOLOGY_UNSUPPORTED",),
        hidden_orphan=hidden,
    )
    transition = owner.consume_reintegrate_update(OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity,
            OperationTerminalStatus.RETURNED,
            payload=aborted,
        ),
    ))
    assert "PREPARED_CAPSULE_MISS:SOURCE_TOPOLOGY_UNSUPPORTED" in transition.notice
    assert hidden in transition.notice
    assert transition.reintegrate_successor is None


def test_average_pending_retry_cancel_and_stamp_remain_exact() -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(21)
    slot._identity = identity
    owner._average = AverageOperationState(
        identity, 5, "/detached/average.nexus", "entry"
    )
    pending = OperationPending(
        identity, 2, "source-cleanup", "source close retained"
    )
    transition = owner.consume_average_update(
        OperationUpdate(identity, pending=pending),
        current_intent_revision=5,
    )
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert owner.average_pending is pending
    assert owner.retry_average()
    assert slot.retry_calls == [(identity, pending)]
    assert owner.average_pending is None
    owner.observe_stamp(
        OperationContextStamp(5, "browse", 8), intent_revision=5
    )
    assert slot.observed == [OperationContextStamp(5)]
    assert owner.cancel_average()
    assert slot.cancel_calls == [identity]


def test_average_freezes_project_root_before_worker_dispatch() -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(22)
    slot.next_identity = identity
    configuration = RunIntent(project_root="/project").freeze()
    assert owner.begin_average(
        configuration,
        "/detached/average.nexus",
        revision=3,
    ) is identity
    assert owner.average_state is not None
    assert owner.average_state.source_root == "/project"

    refused, refused_slot = _owner_with_slot()
    refused_slot.next_identity = OperationIdentity(23)
    malformed = replace(configuration, project_root="relative/project")
    assert refused.begin_average(
        malformed,
        "/detached/average.nexus",
        revision=3,
    ) is None
    assert refused_slot.average_calls == []


def test_average_stale_terminal_never_reloads_and_lost_owner_retires_state(
) -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(31)
    target = "/detached/average.nxs"
    slot._identity = identity
    owner._average = AverageOperationState(identity, 4, target, "entry")
    result = _average_result(target)
    transition = owner.consume_average_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=result
            ),
            stale=True,
        ),
        current_intent_revision=4,
    )
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert transition.average_reload is None
    assert transition.request_catalog
    assert owner.average_state is None

    owner._average = AverageOperationState(identity, 4, target, "entry")
    lost = owner.consume_lost_owner(identity)
    assert lost.effect is WorkspaceRefreshEffect.CONTROLS
    assert owner.average_state is None


def test_committed_average_reload_transfers_without_dual_operation_ownership(
) -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(33)
    target = "/detached/average.nxs"
    slot._identity = identity
    state = AverageOperationState(identity, 4, target, "entry")
    owner._average = state
    result = _average_result(target)
    transition = owner.consume_average_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=result
            ),
        ),
        current_intent_revision=4,
    )
    directive = transition.average_reload
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert directive is not None
    assert owner.average_state is None
    slot._identity = None
    assert not owner.busy


@pytest.mark.parametrize(
    ("status", "payload"),
    (
        (OperationTerminalStatus.CANCELLED, None),
        (OperationTerminalStatus.RETURNED, None),
        (
            OperationTerminalStatus.RETURNED,
            replace(
                _average_result("/detached/average.nxs"),
                disposition="REFUSED",
                committed_labels=(),
                finite_counts=None,
                diagnostic_code="AVERAGE_REFUSED",
                diagnostic="refused",
                h23_phase=None,
                commit_identity=None,
            ),
        ),
    ),
)
def test_non_science_average_terminals_are_controls_only(
    status: OperationTerminalStatus, payload: object,
) -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(34)
    target = "/detached/average.nxs"
    slot._identity = identity
    owner._average = AverageOperationState(identity, 4, target, "entry")
    transition = owner.consume_average_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, status, payload=payload,
            ),
        ),
        current_intent_revision=4,
    )
    assert transition.effect is WorkspaceRefreshEffect.CONTROLS
    assert transition.average_reload is None
    assert owner.average_state is None


def test_close_preserves_active_state_until_exact_slot_cleanup() -> None:
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(41)
    slot._identity = identity
    owner._average = AverageOperationState(
        identity, 1, "/detached/average.nexus", "entry"
    )
    slot.close_receipts = [
        OperationCleanupReceipt(
            identity,
            CleanupStatus.CLEANUP_PENDING,
            cancel_accepted=True,
            worker_identity=7,
        ),
        OperationCleanupReceipt(
            identity,
            CleanupStatus.CLEANED,
            cancel_accepted=True,
            worker_identity=7,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.CANCELLED
            ),
        ),
    ]
    assert owner.close().cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert owner.average_identity is identity
    assert owner.close().cleanup_status is CleanupStatus.CLEANED
    assert owner.average_state is None
