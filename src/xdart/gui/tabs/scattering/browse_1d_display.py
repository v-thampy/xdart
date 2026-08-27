"""Failure-atomic, bounded display copies for cache-backed Browse traces."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from xrd_tools.core import Axis, FrameView

from .browse_1d_projection import Browse1DBorrowBundle, Browse1DProjectionStatus
from .browse_1d_target_plan import (
    Browse1DRuntimeProjection,
    Browse1DTargetPlan,
    MAX_BROWSE_1D_DETACHED_ROOTS,
)
from .display_values import StandardDisplayPayload
from .shell_values import BrowseTraceSnapshot


MAX_BROWSE_1D_DETACHED_BYTES = 64 << 20


class Browse1DDisplayRefusal(RuntimeError):
    """The borrowed projection could not become a complete display snapshot."""


class Browse1DReleaseDebt(RuntimeError):
    """Exact retry custody for a bundle that did not fully release."""

    def __init__(self, bundle: Browse1DBorrowBundle, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.bundle = bundle


@dataclass(frozen=True, slots=True, eq=False)
class DetachedBrowse1DProjection:
    """Display-owned rows returned only after all cache borrows retire."""

    payloads: tuple[StandardDisplayPayload, ...]
    trace_snapshot: BrowseTraceSnapshot
    plan: Browse1DTargetPlan

    def __post_init__(self) -> None:
        if (
            type(self.payloads) is not tuple
            or not self.payloads
            or type(self.plan) is not Browse1DTargetPlan
            or self.plan.logical_frames is not self.trace_snapshot.logical_frames
            or self.plan.display_targets is not self.trace_snapshot.display_frames
            or self.plan.logical_positions is not self.trace_snapshot.logical_positions
            or len(self.payloads) != len(self.trace_snapshot.display_frames)
            or any(
                type(payload) is not StandardDisplayPayload
                or payload.frame_key is not frame
                for payload, frame in zip(
                    self.payloads,
                    self.trace_snapshot.display_frames,
                    strict=True,
                )
            )
        ):
            raise TypeError("detached Browse 1-D projection is invalid")


def _payload_arrays(
    payload: StandardDisplayPayload,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    view = payload.view
    axis = view.axis_1d
    if (
        type(view) is not FrameView
        or type(axis) is not Axis
        or axis.values is None
        or view.intensity_1d is None
    ):
        raise Browse1DDisplayRefusal("Browse 1-D payload is incomplete")
    arrays = (axis.values, view.intensity_1d, view.sigma_1d)
    if any(
        value is not None
        and (type(value) is not np.ndarray or value.ndim != 1)
        for value in arrays
    ):
        raise Browse1DDisplayRefusal(
            "Browse 1-D display array is not an exact one-dimensional ndarray"
        )
    return arrays


def _preflight_copy_sources(
    payloads: tuple[StandardDisplayPayload, ...],
) -> dict[int, np.ndarray]:
    """Admit every unique exact source before allocating packed storage."""

    sources: dict[int, np.ndarray] = {}
    copied_bytes = 0
    for payload in payloads:
        for source in _payload_arrays(payload):
            if source is None:
                continue
            identity = id(source)
            held = sources.get(identity)
            if held is not None:
                if held is not source:
                    raise Browse1DDisplayRefusal(
                        "Browse 1-D source identity was reused"
                    )
                continue
            sources[identity] = source
            copied_bytes += int(source.size) * np.dtype(np.float64).itemsize
            if copied_bytes > MAX_BROWSE_1D_DETACHED_BYTES:
                raise Browse1DDisplayRefusal(
                    "Browse 1-D detached display budget exceeded"
                )
    # All admitted sources are packed into one immutable allocation.  Keep the
    # root admission explicit and pre-allocation so a configured zero ceiling
    # still fails closed without allocating.
    copied_root_count = 1 if sources else 0
    if copied_root_count > MAX_BROWSE_1D_DETACHED_ROOTS:
        raise Browse1DDisplayRefusal(
            "Browse 1-D detached display budget exceeded"
        )
    return sources


def _packed_display_roots(
    sources: dict[int, np.ndarray],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Copy every exact source once into one immutable display-owned root."""

    total_values = sum(int(source.size) for source in sources.values())
    packed = np.empty(total_values, dtype=np.float64)
    spans: dict[int, tuple[np.ndarray, int, int]] = {}
    cursor = 0
    for identity, source in sources.items():
        stop = cursor + int(source.size)
        np.copyto(packed[cursor:stop], source, casting="unsafe")
        spans[identity] = (source, cursor, stop)
        cursor = stop
    packed.setflags(write=False)
    if packed.base is not None or packed.flags.writeable:
        raise Browse1DDisplayRefusal(
            "Browse 1-D packed display copy did not own its root"
        )
    roots = {
        identity: (source, packed[start:stop])
        for identity, (source, start, stop) in spans.items()
    }
    if any(
        copied.base is not packed or copied.flags.writeable
        for _source, copied in roots.values()
    ):
        raise Browse1DDisplayRefusal(
            "Browse 1-D packed display views are not immutable"
        )
    return roots


def _copied_root(
    source: np.ndarray,
    roots: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    held, copied = roots[id(source)]
    if held is not source:
        raise Browse1DDisplayRefusal("Browse 1-D copied source identity drifted")
    return copied


def _copy_payload(
    payload: StandardDisplayPayload,
    roots: dict[int, tuple[np.ndarray, np.ndarray]],
) -> StandardDisplayPayload:
    view = payload.view
    axis = view.axis_1d
    axis_source, intensity_source, sigma_source = _payload_arrays(payload)
    axis_values = _copied_root(axis_source, roots)
    intensity = _copied_root(intensity_source, roots)
    sigma = (
        None
        if sigma_source is None
        else _copied_root(sigma_source, roots)
    )
    copied_axis = Axis(axis.label, axis.unit, axis.log, axis_values)
    copied_view = replace(
        view,
        axis_1d=copied_axis,
        intensity_1d=intensity,
        sigma_1d=sigma,
    )
    if (
        copied_axis.values is not axis_values
        or copied_view.intensity_1d is not intensity
        or sigma is not None and copied_view.sigma_1d is not sigma
    ):
        raise Browse1DDisplayRefusal("Browse 1-D display roots were copied again")
    return replace(payload, view=copied_view)


def detach_browse_1d_projection(
    runtime: Browse1DRuntimeProjection,
) -> DetachedBrowse1DProjection:
    """Copy planned rows, then retire every borrow before returning to Qt."""

    if (
        type(runtime) is not Browse1DRuntimeProjection
        or runtime.status is not Browse1DProjectionStatus.COMPLETE
        or runtime.plan is None
        or type(runtime.borrow_bundle) is not Browse1DBorrowBundle
    ):
        raise Browse1DDisplayRefusal("Browse 1-D runtime projection is not complete")
    bundle = runtime.borrow_bundle
    copied: tuple[StandardDisplayPayload, ...] | None = None
    copy_error: BaseException | None = None
    try:
        sources = _preflight_copy_sources(runtime.payloads)
        roots = _packed_display_roots(sources)
        copied = tuple(
            _copy_payload(payload, roots) for payload in runtime.payloads
        )
    except BaseException as error:
        copy_error = error
    try:
        bundle.release()
    except BaseException as error:
        raise Browse1DReleaseDebt(
            bundle,
            f"{type(error).__name__}: {error}"[:512],
        ) from None
    if copy_error is not None or copied is None:
        raise Browse1DDisplayRefusal(
            f"{type(copy_error).__name__}: {copy_error}"[:512]
        ) from None
    try:
        snapshot = BrowseTraceSnapshot(
            runtime.plan.logical_frames,
            runtime.plan.display_targets,
            runtime.plan.logical_positions,
            runtime.plan.logical_epochs,
            runtime.plan.plot_mode,
            runtime.plan.waterfall_active,
            runtime.plan.stacked_options_applied,
        )
        return DetachedBrowse1DProjection(copied, snapshot, runtime.plan)
    except BaseException as error:
        raise Browse1DDisplayRefusal(
            f"{type(error).__name__}: {error}"[:512]
        ) from None


def release_runtime_borrows(runtime: Browse1DRuntimeProjection) -> None:
    """Retire cleanup custody carried by a non-complete runtime result."""

    if type(runtime) is not Browse1DRuntimeProjection:
        raise TypeError("Browse 1-D runtime projection must be exact")
    bundle = runtime.borrow_bundle
    if bundle is None:
        return
    try:
        bundle.release()
    except BaseException as error:
        raise Browse1DReleaseDebt(
            bundle,
            f"{type(error).__name__}: {error}"[:512],
        ) from None


def prepare_browse_1d_display(
    runtime: Browse1DRuntimeProjection,
) -> DetachedBrowse1DProjection | None:
    """Return one paintable copy, or the explicit preserve-current outcome."""

    if type(runtime) is not Browse1DRuntimeProjection:
        raise TypeError("Browse 1-D runtime projection must be exact")
    if runtime.status is Browse1DProjectionStatus.COMPLETE:
        return detach_browse_1d_projection(runtime)
    release_runtime_borrows(runtime)
    if runtime.status is Browse1DProjectionStatus.REFUSED:
        raise Browse1DDisplayRefusal(
            runtime.diagnostic or "Browse 1-D display was refused"
        )
    return None


__all__ = [
    "Browse1DDisplayRefusal",
    "Browse1DReleaseDebt",
    "DetachedBrowse1DProjection",
    "MAX_BROWSE_1D_DETACHED_BYTES",
    "MAX_BROWSE_1D_DETACHED_ROOTS",
    "detach_browse_1d_projection",
    "prepare_browse_1d_display",
    "release_runtime_borrows",
]
