"""Composition boundary for one exact, non-empty headless ``Scan``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from xrd_tools.reduction import (
    FrameBackgroundPlan,
    ReductionPlan,
    Scan,
    StrictPolicy,
    bind_dynamic_output_sink,
)

from .dynamic_accounting import DynamicRunAccounting
from .frame_record_store import FrameRecordStore
from .policy import (
    SessionPolicy,
    SessionResourceAllocation,
    requirements_from,
    resolve_session_policy,
)
from .scan_session import ScanSession, required_result_modes


class DynamicXyeReceiptBoundaryRequired(TypeError):
    """A dynamic XYE target lacks the accepted transaction receipt owner."""


def _target_maps(
    plan: ReductionPlan,
    *,
    nexus_target: str | None,
    xye_target: str | None,
) -> tuple[
    Mapping[Any, tuple[str, ...]] | None,
    Mapping[Any, tuple[str, ...]] | None,
]:
    targets: dict[Any, tuple[str, ...]] = {}
    store: dict[Any, tuple[str, ...]] = {}
    for mode in required_result_modes(plan):
        applicable = []
        if nexus_target:
            applicable.append(nexus_target)
        if xye_target and mode.kind == "1d":
            applicable.append(xye_target)
        if not applicable:
            return None, None
        targets[mode] = tuple(applicable)
        store[mode] = (nexus_target,) if nexus_target else ()
    return (targets, store) if targets else (None, None)


def _validate_policy(
    scan: Scan,
    plan: ReductionPlan,
    policy: SessionPolicy | None,
    background_plan: FrameBackgroundPlan | None,
) -> None:
    if policy is None:
        return
    if type(policy) is not SessionPolicy:
        raise TypeError("policy must be an exact SessionPolicy or None")
    if type(policy.allocation) is not SessionResourceAllocation:
        raise TypeError("policy must carry an exact SessionResourceAllocation")

    image = scan.frames[0].image
    descriptor = SimpleNamespace(
        frame_shape=tuple(getattr(image, "shape", ())),
        dtype=getattr(image, "dtype", None),
    )
    supplied = policy.allocation.requirements
    terms = (
        supplied.background_bytes,
        supplied.resolver_background_bytes,
        supplied.worker_background_bytes,
        supplied.background_binding_bytes,
    )
    if background_plan is None and terms != (0, 0, 0, 0):
        raise ValueError("active allocation requires an exact Background plan")
    background = (
        FrameBackgroundPlan() if background_plan is None else background_plan
    )
    if type(background) is not FrameBackgroundPlan:
        raise TypeError("Background plan must be exact")
    pixels = int(np.prod(descriptor.frame_shape))
    expected = (
        (0, 0, 0, 0)
        if background.mode == "None"
        else (
            8 * pixels,
            25 * pixels if background.mode == "Series Average" else 8 * pixels,
            8 * pixels,
            64 * 1024**2,
        )
    )
    if terms != expected:
        raise ValueError("explicit allocation has invalid Background resource terms")
    actual = requirements_from(
        descriptor,
        plan,
        background_bytes=supplied.background_bytes,
        resolver_background_bytes=supplied.resolver_background_bytes,
        worker_background_bytes=supplied.worker_background_bytes,
        background_binding_bytes=supplied.background_binding_bytes,
    )
    validated = resolve_session_policy(
        actual,
        allocation=policy.allocation,
        flush=policy.flush,
        env={},
    )
    if validated.allocation is not policy.allocation:
        raise RuntimeError("explicit allocation validator replaced allocation identity")


def open_headless_scan_session(
    scan: Scan,
    plan: ReductionPlan,
    *,
    executor: Any = None,
    cancel_token: Any = None,
    gi_freeze_mode: str | None = None,
    sink: Any = None,
    inflight_max: int | None = None,
    record_store: FrameRecordStore | None = None,
    record_store_persisted_on_write: bool = False,
    nexus_target: str | None = None,
    xye_target: str | None = None,
    policy: SessionPolicy | None = None,
    accounting: DynamicRunAccounting | None = None,
    xye_receipt_boundary: Any = None,
    background_plan: FrameBackgroundPlan | None = None,
    dynamic_nexus_checkpoint: bool = False,
    dynamic_nexus_checkpoint_threshold: int | None = None,
) -> ScanSession:
    """Open the accepted session graph over one exact non-empty ``Scan``."""
    if type(scan) is not Scan:
        raise TypeError("headless session requires an exact Scan")
    if not scan.frames:
        raise ValueError("cannot open a session without frames")
    _validate_policy(scan, plan, policy, background_plan)

    dynamic_accounting = None
    stage_accounting = None
    if accounting is not None:
        if type(accounting) is not DynamicRunAccounting:
            raise TypeError(
                "non-None live accounting must be DynamicRunAccounting authority"
            )
        try:
            binding = bind_dynamic_output_sink(sink)
        except TypeError as error:
            raise DynamicXyeReceiptBoundaryRequired(
                "dynamic output sink is outside the bound P0 supported envelope"
            ) from error
        transactional_xye = binding.transactional_xye_sink
        if transactional_xye is None:
            if xye_target is not None or xye_receipt_boundary is not None:
                raise DynamicXyeReceiptBoundaryRequired(
                    "dynamic XYE target requires the exact transactional sink"
                )
        elif xye_target != transactional_xye.canonical_target:
            raise DynamicXyeReceiptBoundaryRequired(
                "dynamic XYE target must exactly match its transaction owner"
            )
        elif (
            xye_receipt_boundary is not None
            and xye_receipt_boundary is not transactional_xye
        ):
            raise DynamicXyeReceiptBoundaryRequired(
                "dynamic XYE receipt boundary must be the exact bound sink"
            )

        pre_targets, _ = _target_maps(
            plan, nexus_target=nexus_target, xye_target=xye_target
        )
        nexus = binding.nexus_sink
        if nexus is not None:
            if nexus.allow_unbound_same_run and nexus.same_run_intent is None:
                raise ValueError(
                    "dynamic ScanSession refuses unbound same-run adoption"
                )
            if nexus.flush_every is not None:
                raise ValueError("dynamic Nexus sink requires flush_every=None")
            if nexus_target != f"nexus:{nexus.path}":
                raise ValueError(
                    "dynamic Nexus target must exactly match every required mode"
                )
        if transactional_xye is not None:
            declared = (
                None
                if pre_targets is None
                else {
                    mode: frozenset(pre_targets.get(mode, ()))
                    for mode in accounting.ledger.required_modes
                }
            )
            if declared != dict(accounting.ledger.targets_by_mode):
                raise DynamicXyeReceiptBoundaryRequired(
                    "dynamic XYE targets must exactly match the bound graph"
                )
        elif nexus is not None:
            expected = frozenset((f"nexus:{nexus.path}",))
            if pre_targets is None or any(
                frozenset(pre_targets.get(mode, ())) != expected
                for mode in accounting.ledger.required_modes
            ):
                raise ValueError(
                    "dynamic Nexus target must exactly match every required mode"
                )
        dynamic_accounting = accounting
        stage_accounting = accounting.ledger
        sink = binding.sink

    targets, store_targets = _target_maps(
        plan, nexus_target=nexus_target, xye_target=xye_target
    )
    return ScanSession(
        plan,
        scan,
        sink=sink,
        executor=executor,
        inflight_max=inflight_max,
        gi_freeze_mode=gi_freeze_mode,
        cancel_token=cancel_token,
        clear_frame_images=True,
        record_store=record_store,
        record_store_persisted_on_write=record_store_persisted_on_write,
        targets_by_mode=targets,
        store_targets_by_mode=store_targets,
        policy=policy,
        accounting=stage_accounting,
        dynamic_accounting=dynamic_accounting,
        dynamic_nexus_checkpoint=dynamic_nexus_checkpoint,
        dynamic_nexus_checkpoint_threshold=dynamic_nexus_checkpoint_threshold,
        strict=StrictPolicy.graceful(),
    )


__all__ = ["DynamicXyeReceiptBoundaryRequired", "open_headless_scan_session"]
