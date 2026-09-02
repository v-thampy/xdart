"""Qt-free cache-backed Browse 1-D projection.

This module performs no HDF5 or filesystem access and never mutates a
publication or record store.  It turns exact scalar-catalog facts plus exact
non-evictable cache borrows into display payloads.  The returned linear bundle
is the sole eviction custody for every array installed in those payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdart.modules.display_context import (
    BrowseContext,
    CommitGate,
    ContextKind,
    DisplaySelection,
)
from xdart.modules.frame_publication import FramePublication, PublicationStore
from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
from xrd_tools.io import Browse1DCache, FrameScalarCatalog
from xrd_tools.io.browse_1d_cache import Browse1DBorrow
from xrd_tools.io.output_transaction import TargetSnapshot

from .browse_1d_hydration import (
    Browse1DHydrationLane,
    Browse1DLabelInventory,
    Browse1DModeInventory,
)
from .browse_values import canonical_browse_source_identity
from .display_values import DisplayFrameKey, StandardDisplayPayload
from .shell_values import FrameNavigationProjection


_MAX_DIAGNOSTIC_CHARS = 512


class Browse1DProjectionStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    REFUSED = "refused"


class Browse1DBorrowBundle:
    """Linear, retryable release custody for one all-or-none projection."""

    __slots__ = ("_borrows", "_offset", "_released")

    def __init__(self, borrows: tuple[Browse1DBorrow, ...]) -> None:
        if (
            type(borrows) is not tuple
            or any(type(item) is not Browse1DBorrow for item in borrows)
            or len({id(item) for item in borrows}) != len(borrows)
            or any(item.released for item in borrows)
        ):
            raise TypeError("Browse 1-D borrow bundle is invalid")
        self._borrows = borrows
        self._offset = 0
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def remaining(self) -> int:
        return 0 if self._released else len(self._borrows) - self._offset

    def release(self) -> None:
        if self._released:
            return
        while self._offset < len(self._borrows):
            borrowed = self._borrows[self._offset]
            try:
                borrowed.release()
            except BaseException:
                # A wrapper may interrupt after the exact borrow retired.
                # Advance only on the borrow's own durable terminal fact.
                if borrowed.released:
                    self._offset += 1
                raise
            if not borrowed.released:
                raise RuntimeError("Browse 1-D borrow did not retire")
            self._offset += 1
        self._borrows = ()
        self._released = True

    def __enter__(self) -> "Browse1DBorrowBundle":
        if self._released:
            raise RuntimeError("Browse 1-D borrow bundle is released")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()

    def __copy__(self):
        raise TypeError("Browse 1-D borrow bundles cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("Browse 1-D borrow bundles cannot be copied")

    def __reduce__(self):
        raise TypeError("Browse 1-D borrow bundles cannot be serialized")

    def __reduce_ex__(self, _protocol):
        raise TypeError("Browse 1-D borrow bundles cannot be serialized")


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DProjectionOutcome:
    status: Browse1DProjectionStatus
    payloads: tuple[StandardDisplayPayload, ...] = ()
    borrow_bundle: Browse1DBorrowBundle | None = None
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.status) is not Browse1DProjectionStatus
            or type(self.payloads) is not tuple
            or any(type(item) is not StandardDisplayPayload for item in self.payloads)
            or self.borrow_bundle is not None
            and type(self.borrow_bundle) is not Browse1DBorrowBundle
            or type(self.diagnostic) is not str
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARS
        ):
            raise TypeError("Browse 1-D projection outcome is invalid")
        if self.status is Browse1DProjectionStatus.COMPLETE:
            if not self.payloads or self.borrow_bundle is None:
                raise ValueError("complete Browse projection requires payload custody")
        elif self.payloads:
            raise ValueError("incomplete Browse projection cannot expose payloads")


class _Incomplete(RuntimeError):
    pass


class _Refused(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class _QualifiedHeavy:
    axis_2d_x: Axis | None
    axis_2d_y: Axis | None
    intensity_2d: object | None
    sigma_2d: object | None
    two_d_kind: TwoDKind
    raw: object | None
    thumbnail: object | None


@dataclass(frozen=True, slots=True, eq=False)
class _ProjectionScope:
    context: BrowseContext
    hydration: Browse1DHydrationLane
    catalog: FrameScalarCatalog
    cache: Browse1DCache
    gate: CommitGate
    epoch: int
    artifact: str
    entry: str
    snapshot: TargetSnapshot
    selection: DisplaySelection
    navigation: FrameNavigationProjection
    targets: tuple[DisplayFrameKey, ...]
    generation: int


def _diagnostic(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:_MAX_DIAGNOSTIC_CHARS]


def _scope_is_live(scope: _ProjectionScope) -> bool:
    context = scope.context
    return bool(
        context.scalar_catalog is scope.catalog
        and context.browse_1d_cache is scope.cache
        and context.commit_gate is scope.gate
        and context.commit_epoch == scope.epoch
        and context.requested_path == scope.artifact
        and context.target_entry == scope.entry
        and context.target_snapshot is scope.snapshot
        and scope.catalog.artifact_path == scope.artifact
        and scope.catalog.entry == scope.entry
        and scope.catalog.labels is context.frame_ids
        and scope.catalog.labels is context.loaded_labels
        and scope.hydration.context is context
        and scope.selection.display_generation == scope.generation
        and scope.selection.kind is ContextKind.BROWSE
        and scope.selection.names(context)
        and scope.selection.owner == context.hydration_owner
        and context.loaded
        and not context.invalidated
        and not context.released
        and not scope.gate.cancelled
    )


def _admit_scope(
    context: object,
    hydration: object,
    selection: object,
    current_selection: object,
    navigation: object,
    targets: object,
) -> _ProjectionScope:
    if (
        type(context) is not BrowseContext
        or type(hydration) is not Browse1DHydrationLane
        or type(selection) is not DisplaySelection
        or selection is not current_selection
        or type(navigation) is not FrameNavigationProjection
        or type(targets) is not tuple
        or not targets
        or any(type(item) is not DisplayFrameKey for item in targets)
        or len({id(item) for item in targets}) != len(targets)
        or type(context.scalar_catalog) is not FrameScalarCatalog
        or type(context.browse_1d_cache) is not Browse1DCache
        or type(context.commit_gate) is not CommitGate
        or type(context.commit_epoch) is not int
        or context.commit_epoch < 1
        or type(context.target_snapshot) is not TargetSnapshot
        or not context.target_snapshot.exists
        or type(context.requested_path) is not str
        or not context.requested_path
        or type(context.target_entry) is not str
        or not context.target_entry
    ):
        raise _Refused("Browse 1-D projection scope is malformed")
    scope = _ProjectionScope(
        context,
        hydration,
        context.scalar_catalog,
        context.browse_1d_cache,
        context.commit_gate,
        context.commit_epoch,
        context.requested_path,
        context.target_entry,
        context.target_snapshot,
        selection,
        navigation,
        targets,
        selection.display_generation,
    )
    if not _scope_is_live(scope):
        raise _Refused("Browse 1-D projection scope is stale")
    navigation_ids = {id(item) for item in navigation.frames}
    run_identity = targets[0].run_identity
    previous = 0
    for frame in targets:
        if (
            id(frame) not in navigation_ids
            or frame.run_identity is not run_identity
            or frame.source_scan != context.scan_key
            or frame.artifact != scope.artifact
            or frame.work_ordinal <= previous
            or frame.work_ordinal > len(scope.catalog.labels)
            or scope.catalog.labels[frame.work_ordinal - 1]
            != frame.local_frame_label
        ):
            raise _Refused("Browse 1-D target is not runtime-owned")
        previous = frame.work_ordinal
    if navigation.current is not None and id(navigation.current) not in navigation_ids:
        raise _Refused("Browse 1-D current frame is foreign")
    return scope


def _inventory_is_same(
    first: Browse1DLabelInventory,
    second: Browse1DLabelInventory,
) -> bool:
    if (
        type(first) is not Browse1DLabelInventory
        or type(second) is not Browse1DLabelInventory
        or first.ordinal != second.ordinal
        or first.label != second.label
        or len(first.modes) != len(second.modes)
    ):
        return False
    for left, right in zip(first.modes, second.modes):
        if (
            left.mode != right.mode
            or len(left.keys) != len(right.keys)
            or any(
                a.frame != b.frame
                or a.label != b.label
                or a.name != b.name
                for a, b in zip(left.keys, right.keys)
            )
        ):
            return False
    return True


def _active_mode(
    inventory: Browse1DLabelInventory,
    scalar_row,
) -> Browse1DModeInventory | None:
    if tuple(item.mode for item in inventory.modes) != scalar_row.modes_1d:
        raise _Refused("Browse 1-D inventory mode coverage changed")
    active = scalar_row.active_mode_1d
    if not scalar_row.modes_1d:
        if active is not None or inventory.modes:
            raise _Refused("Browse empty 1-D mode inventory is inconsistent")
        return None
    if type(active) is not str or not active:
        raise _Refused("Browse 1-D active mode is missing")
    result = inventory.mode(active)
    if result is None:
        raise _Refused("Browse 1-D active mode left its inventory")
    return result


def _axis_descriptor(catalog: FrameScalarCatalog, mode: str):
    matches = tuple(item for item in catalog.axes_1d if item[0] == mode)
    if len(matches) != 1:
        raise _Refused("Browse 1-D axis descriptor is not exact")
    return matches[0]


def _exact_record_view(record: FrameRecord, scalar_row, dimension: str):
    if dimension == "1d":
        results = record.results_1d
        scalar_active = scalar_row.active_mode_1d
        record_active = record.active_mode_1d
        getter = record.view_1d
    else:
        results = record.results_2d
        scalar_active = scalar_row.active_mode_2d
        record_active = record.active_mode_2d
        getter = record.view_2d
    if not results:
        return None
    if (
        type(scalar_active) is not str
        or not scalar_active
        or type(record_active) is not str
        or record_active != scalar_active
    ):
        raise _Refused(
            f"Browse sparse active {dimension} mode changed"
        )
    view = getter(scalar_active)
    if (
        type(view) is not FrameView
        or type(view.label) is not int
        or view.label != scalar_row.label
    ):
        raise _Refused(
            f"Browse sparse active {dimension} view changed ownership"
        )
    if dimension == "1d" and not view.has_1d:
        raise _Refused("Browse sparse active 1-D view is incomplete")
    return view


def _two_d_completeness(view: FrameView) -> bool:
    """Return complete-vs-shell; reject a claimed impossible 2-D payload."""

    if view.intensity_2d is None:
        if view.sigma_2d is not None:
            raise _Refused("Browse sparse 2-D shell has uncertainty pixels")
        return False
    if (
        not view.has_2d
        or type(view.axis_2d_x) is not Axis
        or type(view.axis_2d_y) is not Axis
        or view.axis_2d_x.values is None
        or view.axis_2d_y.values is None
    ):
        raise _Refused("Browse sparse 2-D payload is incomplete")
    return True


def _detector_source_is_exact(view: FrameView, scalar_row) -> bool:
    return bool(
        type(view.source_path) is str
        and view.source_path
        and type(scalar_row.source_path) is str
        and scalar_row.source_path
        and view.source_path == scalar_row.source_path
        and type(view.source_frame_index) is int
        and type(scalar_row.source_frame_index) is int
        and view.source_frame_index == scalar_row.source_frame_index
    )


def _current_heavy(scope: _ProjectionScope, frame: DisplayFrameKey, scalar_row):
    if frame is not scope.navigation.current:
        return None
    store = scope.context.publication_store
    if type(store) is not PublicationStore:
        raise _Refused("Browse sparse publication store is foreign")
    publication = store.get(frame.local_frame_label)
    if publication is None:
        return None
    if type(publication) is not FramePublication:
        raise _Refused("Browse sparse publication is foreign")
    expected_source = canonical_browse_source_identity(
        scalar_row,
        scope.artifact,
        source_base=scope.catalog.source_base,
        source_root=scope.context.load_request.source_root,
    )
    record = publication.record
    view = publication.view
    if (
        type(scalar_row.label) is not int
        or type(view) is not FrameView
        or type(view.label) is not int
        or type(record) is not FrameRecord
        or type(record.label) is not int
        or view.label != scalar_row.label
        or record.label != scalar_row.label
        or view.source_path is not None
        and (type(view.source_path) is not str or not view.source_path)
        or view.source_frame_index is not None
        and (
            type(view.source_frame_index) is not int
            or view.source_frame_index < 0
        )
        or type(publication.source_identity) is not str
        or publication.source_identity != expected_source
        or canonical_browse_source_identity(
            view,
            scope.artifact,
            source_base=scope.catalog.source_base,
            source_root=scope.context.load_request.source_root,
        )
        != expected_source
        or type(publication.scan_key) is not str
        or publication.scan_key != scope.context.scan_key
    ):
        raise _Refused("Browse sparse publication changed ownership")
    active_1d_view = _exact_record_view(record, scalar_row, "1d")
    active_2d_view = _exact_record_view(record, scalar_row, "2d")
    witnesses = tuple(
        item for item in (active_1d_view, active_2d_view)
        if item is not None
    )
    if any(
        item.raw is not view.raw
        or item.thumbnail is not view.thumbnail
        for item in witnesses
    ):
        raise _Refused("Browse sparse detector payload changed identity")

    top_complete = _two_d_completeness(view)
    record_complete = (
        False
        if active_2d_view is None
        else _two_d_completeness(active_2d_view)
    )
    if top_complete or record_complete:
        active_2d = scalar_row.active_mode_2d
        kinds = dict(scalar_row.two_d_kinds)
        if (
            not top_complete
            or not record_complete
            or type(active_2d) is not str
            or kinds.get(active_2d) is not view.two_d_kind
            or active_2d_view.two_d_kind is not view.two_d_kind
            or view.axis_2d_x is not active_2d_view.axis_2d_x
            or view.axis_2d_y is not active_2d_view.axis_2d_y
            or view.axis_2d_x.values is not active_2d_view.axis_2d_x.values
            or view.axis_2d_y.values is not active_2d_view.axis_2d_y.values
            or view.intensity_2d is not active_2d_view.intensity_2d
            or view.sigma_2d is not active_2d_view.sigma_2d
        ):
            raise _Refused(
                "Browse sparse 2-D publication changed mode or payload"
            )

    raw = view.raw
    thumbnail = view.thumbnail
    if not witnesses:
        if thumbnail is not None and not scalar_row.has_thumbnail:
            thumbnail = None
        if raw is not None and not _detector_source_is_exact(view, scalar_row):
            raw = None
    if not top_complete and raw is None and thumbnail is None:
        return None
    return _QualifiedHeavy(
        view.axis_2d_x if top_complete else None,
        view.axis_2d_y if top_complete else None,
        view.intensity_2d if top_complete else None,
        view.sigma_2d if top_complete else None,
        view.two_d_kind,
        raw,
        thumbnail,
    )


def _two_d_kind(scalar_row, heavy: _QualifiedHeavy | None) -> TwoDKind:
    if heavy is not None and heavy.intensity_2d is not None:
        return heavy.two_d_kind
    active = scalar_row.active_mode_2d
    if active is not None:
        for mode, kind in scalar_row.two_d_kinds:
            if mode == active:
                return kind
    return TwoDKind.Q_CHI


def _is_gi(scalar_row, kind: TwoDKind) -> bool:
    geometry = scalar_row.geometry
    return bool(
        kind is not TwoDKind.Q_CHI
        or geometry is not None and geometry.incident_angle is not None
    )


def _payload(
    scope: _ProjectionScope,
    frame: DisplayFrameKey,
    inventory: Browse1DLabelInventory,
    borrowed: dict[str, Browse1DBorrow],
) -> StandardDisplayPayload:
    scalar_row = scope.catalog.row(frame.local_frame_label)
    if scalar_row is None:
        raise _Refused("Browse scalar row disappeared")
    mode_inventory = _active_mode(inventory, scalar_row)
    axis = None
    intensity = None
    sigma = None
    if mode_inventory is not None:
        mode, label, unit, log = _axis_descriptor(
            scope.catalog, mode_inventory.mode,
        )
        axis_borrow = borrowed.get(mode_inventory.axis.name)
        intensity_borrow = borrowed.get(mode_inventory.intensity.name)
        sigma_borrow = (
            None
            if mode_inventory.sigma is None
            else borrowed.get(mode_inventory.sigma.name)
        )
        if (
            type(axis_borrow) is not Browse1DBorrow
            or type(intensity_borrow) is not Browse1DBorrow
            or mode_inventory.sigma is not None
            and type(sigma_borrow) is not Browse1DBorrow
        ):
            raise _Refused("Browse 1-D borrow set is incomplete")
        axis = Axis(label, unit, log, axis_borrow.array)
        intensity = intensity_borrow.array
        sigma = None if sigma_borrow is None else sigma_borrow.array
        if axis.values is not axis_borrow.array:
            raise _Refused("Browse 1-D axis construction copied its borrow")
    heavy = _current_heavy(scope, frame, scalar_row)
    kind = _two_d_kind(scalar_row, heavy)
    geometry = scalar_row.geometry
    view = FrameView(
        label=scalar_row.label,
        axis_1d=axis,
        intensity_1d=intensity,
        sigma_1d=sigma,
        axis_2d_x=None if heavy is None else heavy.axis_2d_x,
        axis_2d_y=None if heavy is None else heavy.axis_2d_y,
        intensity_2d=None if heavy is None else heavy.intensity_2d,
        sigma_2d=None if heavy is None else heavy.sigma_2d,
        two_d_kind=kind,
        raw=None if heavy is None else heavy.raw,
        thumbnail=None if heavy is None else heavy.thumbnail,
        mask_baked=scalar_row.mask_baked,
        metadata_raw=scalar_row.metadata_raw,
        metadata_numeric=scalar_row.metadata_numeric,
        incident_angle=(None if geometry is None else geometry.incident_angle),
        geometry=geometry,
        source_path=scalar_row.source_path,
        source_frame_index=scalar_row.source_frame_index,
    )
    if intensity is not None and view.intensity_1d is not intensity:
        raise _Refused("Browse 1-D view copied its intensity borrow")
    if sigma is not None and view.sigma_1d is not sigma:
        raise _Refused("Browse 1-D view copied its sigma borrow")
    gi = _is_gi(scalar_row, kind)
    return StandardDisplayPayload(
        scope.generation,
        frame,
        (
            f"Browse · {scope.context.scan_key} · [averaged]"
            if scalar_row.averaged
            else f"Browse · {scope.context.scan_key} · frame {scalar_row.label}"
        ),
        view,
        "browse",
        measurement_mode="GI" if gi else "Standard",
        gi_incidence_motor="Manual" if gi else "",
        gi_resolved_motor="Manual" if gi else "",
        gi_mode_1d=(scalar_row.active_mode_1d or "") if gi else "",
        gi_mode_2d=(scalar_row.active_mode_2d or "") if gi else "",
        averaged=scalar_row.averaged,
    )


def _failure(
    status: Browse1DProjectionStatus,
    error: BaseException,
    borrows: list[Browse1DBorrow],
) -> Browse1DProjectionOutcome:
    bundle = None
    if borrows:
        bundle = Browse1DBorrowBundle(tuple(borrows))
        try:
            bundle.release()
        except BaseException as cleanup_error:
            return Browse1DProjectionOutcome(
                Browse1DProjectionStatus.REFUSED,
                borrow_bundle=bundle,
                diagnostic=_diagnostic(cleanup_error),
            )
        bundle = None
    return Browse1DProjectionOutcome(status, diagnostic=_diagnostic(error))


def project_browse_1d(
    context: object,
    hydration: object,
    selection: object,
    navigation: object,
    targets: object,
    *,
    current_selection: object,
) -> Browse1DProjectionOutcome:
    """Acquire an exact all-or-none Browse 1-D display projection."""

    borrows: list[Browse1DBorrow] = []
    try:
        scope = _admit_scope(
            context, hydration, selection, current_selection,
            navigation, targets,
        )
        terminal_diagnostic = hydration.terminal_diagnostic(
            selection, scope.targets,
        )
        if terminal_diagnostic is not None:
            return Browse1DProjectionOutcome(
                Browse1DProjectionStatus.REFUSED,
                diagnostic=terminal_diagnostic,
            )
        inventories: list[Browse1DLabelInventory] = []
        modes: list[Browse1DModeInventory | None] = []
        for frame in scope.targets:
            inventory = hydration.expected_inventory(
                frame.work_ordinal, frame.local_frame_label,
            )
            if inventory is None:
                raise _Incomplete("Browse 1-D inventory is not hydrated")
            if type(inventory) is not Browse1DLabelInventory:
                raise _Refused("Browse 1-D inventory owner returned a foreign fact")
            scalar_row = scope.catalog.row(frame.local_frame_label)
            if scalar_row is None:
                raise _Refused("Browse scalar target disappeared")
            inventories.append(inventory)
            modes.append(_active_mode(inventory, scalar_row))

        by_target: list[dict[str, Browse1DBorrow]] = []
        for frame, mode_inventory in zip(scope.targets, modes):
            named: dict[str, Browse1DBorrow] = {}
            if mode_inventory is not None:
                for key in mode_inventory.keys:
                    try:
                        borrowed = scope.cache.borrow(
                            frame.work_ordinal,
                            frame.local_frame_label,
                            key.name,
                        )
                    except KeyError as error:
                        raise _Incomplete(
                            "Browse 1-D row was evicted during projection"
                        ) from error
                    if type(borrowed) is not Browse1DBorrow or borrowed.released:
                        raise _Refused("Browse 1-D cache returned a foreign borrow")
                    if not any(held is borrowed for held in borrows):
                        borrows.append(borrowed)
                    if (
                        borrowed.key.frame != frame.work_ordinal
                        or borrowed.key.label != frame.local_frame_label
                        or borrowed.key.name != key.name
                        or key.name in named
                    ):
                        raise _Refused("Browse 1-D cache returned a foreign borrow")
                    named[key.name] = borrowed
            by_target.append(named)

        if not _scope_is_live(scope):
            raise _Refused("Browse 1-D projection scope drifted during acquire")
        for frame, inventory in zip(scope.targets, inventories):
            current = hydration.expected_inventory(
                frame.work_ordinal, frame.local_frame_label,
            )
            if (
                current is None
                or not _inventory_is_same(inventory, current)
            ):
                raise _Refused("Browse 1-D inventory drifted during acquire")

        payloads = tuple(
            _payload(scope, frame, inventory, named)
            for frame, inventory, named in zip(
                scope.targets, inventories, by_target,
            )
        )
        if not _scope_is_live(scope):
            raise _Refused("Browse 1-D projection scope drifted before return")
        bundle = Browse1DBorrowBundle(tuple(borrows))
        return Browse1DProjectionOutcome(
            Browse1DProjectionStatus.COMPLETE,
            payloads,
            bundle,
        )
    except _Incomplete as error:
        return _failure(Browse1DProjectionStatus.INCOMPLETE, error, borrows)
    except BaseException as error:
        return _failure(Browse1DProjectionStatus.REFUSED, error, borrows)


__all__ = [
    "Browse1DBorrowBundle",
    "Browse1DProjectionOutcome",
    "Browse1DProjectionStatus",
    "project_browse_1d",
]
