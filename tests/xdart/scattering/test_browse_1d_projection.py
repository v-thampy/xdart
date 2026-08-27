"""Focused C7a cache-backed Browse 1-D projection oracles."""

from __future__ import annotations

from copy import copy, deepcopy
import pickle

import numpy as np
import pytest


def _readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _scope(
    tmp_path,
    scalar_rows,
    axes,
    *,
    budget=1 << 20,
    retain_1d_on_eviction=False,
):
    from xdart.gui.tabs.scattering.browse_1d_hydration import (
        Browse1DHydrationLane,
    )
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xdart.modules.display_context import BrowseContext, DisplaySelection
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.io import Browse1DCache, FrameScalarCatalog
    from xrd_tools.io.output_transaction import capture_target_snapshot

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "projection.nxs"
    path.write_bytes(b"stable Browse projection artifact")
    catalog = FrameScalarCatalog(
        str(path.resolve()), "entry", tuple(scalar_rows), tuple(axes),
    )
    cache = Browse1DCache(budget)
    publications = PublicationStore(
        max_items=16,
        max_heavy_items=16,
        max_thumbnail_items=16,
        retain_1d_on_eviction=retain_1d_on_eviction,
    )
    request = BrowseLoadRequest("browse-projection", 1, str(path.resolve()))
    context = BrowseContext(
        context_token=request.token,
        load_generation=request.load_generation,
        operation=request,
        requested_path=request.source_path,
        scan_key="scan",
        scan=object(),
        frame=None,
        frame_ids=catalog.labels,
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=publications,
        record_store={},
        scalar_catalog=catalog,
        browse_1d_cache=cache,
        target_entry=catalog.entry,
        loaded_labels=catalog.labels,
        target_snapshot=capture_target_snapshot(path),
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    selection = DisplaySelection.for_context(context, 7)
    identity = RunIdentity(1, "browse-projection")
    frames = tuple(
        DisplayFrameKey(
            identity, context.scan_key, context.requested_path,
            label, ordinal,
        )
        for ordinal, label in enumerate(catalog.labels, 1)
    )
    navigation = FrameNavigationProjection(
        frames, frames[0], frames,
    )
    reader_calls = []

    def reader_bomb(*_args, **_kwargs):
        reader_calls.append(True)
        raise AssertionError("projection attempted HDF access")

    hydration = Browse1DHydrationLane(context, open_reader=reader_bomb)
    return (
        context, selection, navigation, frames, catalog, cache,
        publications, hydration, reader_calls,
    )


def _seed_label(
    hydration,
    cache,
    catalog,
    ordinal,
    label,
    *,
    sigma_modes=(),
    serial=0,
):
    from xrd_tools.core import Axis
    from xrd_tools.io import (
        Frame1DModeRows,
        Frame1DRows,
        browse_1d_row_name,
    )

    scalar_row = catalog.row(label)
    assert scalar_row is not None
    arrays = {}
    modes = []
    for index, descriptor in enumerate(catalog.axes_1d):
        mode, axis_label, axis_unit, axis_log = descriptor
        axis = _readonly([
            0.25 + index, 0.5 + index, 0.75 + index,
        ])
        arrays[browse_1d_row_name(mode, "axis")] = axis
        if mode in scalar_row.modes_1d:
            intensity = _readonly([
                serial * 100 + label,
                serial * 100 + label + 1,
                serial * 100 + label + 2,
            ])
            arrays[browse_1d_row_name(mode, "intensity")] = intensity
            sigma_rows = None
            if mode in sigma_modes:
                sigma = _readonly([0.1, 0.2, 0.3])
                arrays[browse_1d_row_name(mode, "sigma")] = sigma
                sigma_rows = (sigma,)
            modes.append(Frame1DModeRows(
                mode,
                Axis(axis_label, axis_unit, axis_log, axis),
                (label,),
                (intensity,),
                sigma_rows,
            ))
        else:
            modes.append(Frame1DModeRows(
                mode,
                Axis(axis_label, axis_unit, axis_log, axis),
                (),
                (),
                None,
            ))
    rows = Frame1DRows(
        catalog.artifact_path,
        catalog.entry,
        (label,),
        tuple(modes),
        scalar_row.active_mode_1d,
    )
    receipt = cache.begin_store_1d_label(
        catalog, rows, ordinal, label,
    )
    if receipt.operation is not None:
        assert receipt.operation.run() == "accepted"
    with hydration._lock:
        hydration._known_keys[(ordinal, label)] = tuple(receipt.keys)
    return arrays


def _project(scope, targets=None, *, selection=None, current_selection=None):
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        project_browse_1d,
    )

    (
        context, owned_selection, navigation, frames, _catalog_value,
        _cache, _publications, hydration, _reader_calls,
    ) = scope
    selected = owned_selection if selection is None else selection
    current = selected if current_selection is None else current_selection
    return project_browse_1d(
        context,
        hydration,
        selected,
        navigation,
        frames if targets is None else targets,
        current_selection=current,
    )


def _close(scope) -> None:
    cache = scope[5]
    hydration = scope[7]
    assert hydration.release()
    cache.close()


@pytest.mark.parametrize(
    "modes,active,sigma_modes,geometry,expected_measurement",
    (
        (("q",), "q", (), None, "Standard"),
        (
            ("q_ip", "q_oop"),
            "q_oop",
            ("q_oop",),
            "gi",
            "GI",
        ),
    ),
)
def test_active_standard_and_gi_modes_preserve_borrowed_array_identity(
    tmp_path, modes, active, sigma_modes, geometry, expected_measurement,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xrd_tools.core import ViewFrameGeometry
    from xrd_tools.io import FrameScalarRow, browse_1d_row_name

    persisted_geometry = (
        None if geometry is None else ViewFrameGeometry(incident_angle=0.2)
    )
    row = FrameScalarRow(
        1,
        metadata_raw={"motor": 2.5},
        geometry=persisted_geometry,
        source_path="detector.tif",
        source_frame_index=4,
        modes_1d=modes,
        active_mode_1d=active,
    )
    axes = tuple((mode, mode.upper(), "A^-1", False) for mode in modes)
    scope = _scope(tmp_path, (row,), axes)
    arrays = _seed_label(
        scope[7], scope[5], scope[4], 1, 1,
        sigma_modes=sigma_modes,
    )
    store_before = tuple(scope[6]._items.items())

    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.COMPLETE
    assert len(outcome.payloads) == 1
    payload = outcome.payloads[0]
    assert payload.frame_key is scope[3][0]
    assert payload.selection_generation == scope[1].display_generation
    assert payload.measurement_mode == expected_measurement
    assert payload.gi_mode_1d == (active if expected_measurement == "GI" else "")
    assert payload.view.axis_1d.values is arrays[
        browse_1d_row_name(active, "axis")
    ]
    assert payload.view.intensity_1d is arrays[
        browse_1d_row_name(active, "intensity")
    ]
    sigma_name = browse_1d_row_name(active, "sigma")
    assert payload.view.sigma_1d is arrays.get(sigma_name)
    assert payload.view.metadata_raw["motor"] == 2.5
    assert payload.view.geometry is persisted_geometry
    assert payload.view.source_path == "detector.tif"
    assert payload.view.source_frame_index == 4
    expected_borrows = 3 if active in sigma_modes else 2
    assert outcome.borrow_bundle.remaining == expected_borrows
    assert scope[5].outstanding_borrows == expected_borrows
    assert tuple(scope[6]._items.items()) == store_before
    assert scope[8] == []

    outcome.borrow_bundle.release()
    outcome.borrow_bundle.release()
    assert outcome.borrow_bundle.released
    assert scope[5].outstanding_borrows == 0
    _close(scope)


def test_unknown_and_exact_empty_inventory_are_distinct(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xrd_tools.io import FrameScalarRow, browse_1d_row_name

    row = FrameScalarRow(1)
    scope = _scope(tmp_path, (row,), ())
    assert scope[7].expected_inventory(1, 1) is None
    missing = _project(scope)
    assert missing.status is Browse1DProjectionStatus.INCOMPLETE
    assert missing.payloads == () and missing.borrow_bundle is None

    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    inventory = scope[7].expected_inventory(1, 1)
    assert inventory is not None and inventory.modes == ()
    complete = _project(scope)
    assert complete.status is Browse1DProjectionStatus.COMPLETE
    assert complete.payloads[0].view.axis_1d is None
    assert complete.payloads[0].view.intensity_1d is None
    assert complete.borrow_bundle.remaining == 0
    complete.borrow_bundle.release()
    assert scope[5].outstanding_borrows == 0
    _close(scope)


def test_current_merges_only_exact_sparse_heavy_and_noncurrent_stays_light(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.modules.frame_publication import FramePublication, PublicationStore
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow, browse_1d_row_name

    rows = tuple(
        FrameScalarRow(
            label,
            metadata_raw={"sealed": float(label)},
            modes_1d=("q",),
            active_mode_1d="q",
            modes_2d=("map",),
            active_mode_2d="map",
            two_d_kinds=(("map", TwoDKind.Q_CHI),),
        )
        for label in (1, 2)
    )
    scope = _scope(tmp_path, rows, (("q", "Q", "A^-1", False),))
    arrays = tuple(
        _seed_label(scope[7], scope[5], scope[4], ordinal, label)
        for ordinal, label in enumerate((1, 2), 1)
    )
    heavy_arrays = {}
    for label in (1, 2):
        raw = _readonly(np.full((2, 2), label))
        thumb = _readonly(np.full((2, 2), label + 10))
        cake = _readonly(np.full((2, 3), label + 20))
        x = _readonly([1.0, 2.0, 3.0])
        y = _readonly([4.0, 5.0])
        view = FrameView(
            label,
            axis_1d=Axis("foreign", "", False, _readonly([1.0])),
            intensity_1d=_readonly([999.0]),
            axis_2d_x=Axis("Q", "A^-1", False, x),
            axis_2d_y=Axis("chi", "deg", False, y),
            intensity_2d=cake,
            raw=raw,
            thumbnail=thumb,
            metadata_raw={"foreign": 99.0},
        )
        record = FrameRecord(
            label,
            results_2d={"map": view},
            active_mode_2d="map",
        )
        scope[6].upsert(FramePublication(
            view,
            record=record,
            source_identity=canonical_browse_source_identity(
                rows[label - 1], scope[0].requested_path,
            ),
            scan_key=scope[0].scan_key,
        ))
        heavy_arrays[label] = (raw, thumb, cake, x, y)

    calls = []
    real_get = PublicationStore.get

    def get(store, label):
        calls.append(label)
        return real_get(store, label)

    def write_bomb(*_args, **_kwargs):
        raise AssertionError("projection mutated the sparse publication store")

    before = tuple((key, id(value)) for key, value in scope[6]._items.items())
    monkeypatch.setattr(PublicationStore, "get", get)
    monkeypatch.setattr(PublicationStore, "upsert", write_bomb)
    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.COMPLETE
    current, noncurrent = outcome.payloads
    assert calls == [1]
    raw, thumb, cake, x, y = heavy_arrays[1]
    assert current.view.raw is raw
    assert current.view.thumbnail is thumb
    assert current.view.intensity_2d is cake
    assert current.view.axis_2d_x.values is x
    assert current.view.axis_2d_y.values is y
    assert current.view.metadata_raw == {"sealed": 1.0}
    intensity_name = browse_1d_row_name("q", "intensity")
    assert current.view.intensity_1d is arrays[0][intensity_name]
    assert noncurrent.view.raw is None
    assert noncurrent.view.thumbnail is None
    assert noncurrent.view.intensity_2d is None
    assert noncurrent.view.intensity_1d is arrays[1][intensity_name]
    assert tuple((key, id(value)) for key, value in scope[6]._items.items()) == before
    outcome.borrow_bundle.release()
    _close(scope)


@pytest.mark.parametrize(
    "shape",
    (
        "absent",
        "one-d-no-heavy",
        "one-d-detector",
        "retain-one-d-shell",
        "empty-record-thumbnail",
        "empty-record-unproven-raw",
        "empty-record-proven-raw",
    ),
)
def test_optional_current_publication_valid_shapes_do_not_block_cached_1d(
    tmp_path, shape,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow, browse_1d_row_name

    source_path = "detector.tif" if shape == "empty-record-proven-raw" else None
    source_index = 7 if source_path is not None else None
    has_thumbnail = shape in {
        "one-d-detector", "retain-one-d-shell", "empty-record-thumbnail",
    }
    has_2d_mode = shape == "retain-one-d-shell"
    row = FrameScalarRow(
        1,
        source_path=source_path,
        source_frame_index=source_index,
        has_thumbnail=has_thumbnail,
        modes_1d=("q",),
        active_mode_1d="q",
        modes_2d=("map",) if has_2d_mode else (),
        active_mode_2d="map" if has_2d_mode else None,
        two_d_kinds=(("map", TwoDKind.Q_CHI),) if has_2d_mode else (),
    )
    scope = _scope(
        tmp_path,
        (row,),
        (("q", "Q", "A^-1", False),),
        retain_1d_on_eviction=has_2d_mode,
    )
    arrays = _seed_label(scope[7], scope[5], scope[4], 1, 1)
    raw = _readonly([[1.0, 2.0], [3.0, 4.0]])
    thumbnail = _readonly([[5.0, 6.0], [7.0, 8.0]])
    if shape != "absent":
        one_d = shape in {
            "one-d-no-heavy", "one-d-detector", "retain-one-d-shell",
        }
        view = FrameView(
            1,
            axis_1d=(
                Axis("publication", "", False, _readonly([9.0, 10.0]))
                if one_d else None
            ),
            intensity_1d=_readonly([90.0, 100.0]) if one_d else None,
            axis_2d_x=(
                Axis("Q", "A^-1", False, _readonly([1.0, 2.0]))
                if has_2d_mode else None
            ),
            axis_2d_y=(
                Axis("chi", "deg", False, _readonly([3.0, 4.0]))
                if has_2d_mode else None
            ),
            intensity_2d=(
                _readonly([[11.0, 12.0], [13.0, 14.0]])
                if has_2d_mode else None
            ),
            raw=(
                raw
                if shape in {
                    "one-d-detector",
                    "empty-record-unproven-raw",
                    "empty-record-proven-raw",
                }
                else None
            ),
            thumbnail=thumbnail if has_thumbnail else None,
            source_path=source_path,
            source_frame_index=source_index,
        )
        results_1d = {"q": view} if one_d else {}
        results_2d = {"map": view} if has_2d_mode else {}
        record = FrameRecord(
            1,
            results_1d=results_1d,
            results_2d=results_2d,
            active_mode_1d="q" if results_1d else "default",
            active_mode_2d="map" if results_2d else "default",
        )
        scope[6].upsert(FramePublication(
            view,
            record=record,
            source_identity=canonical_browse_source_identity(
                row, scope[0].requested_path,
            ),
            scan_key=scope[0].scan_key,
        ))
        if has_2d_mode:
            assert scope[6].evict_heavy(1)

    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.COMPLETE
    payload = outcome.payloads[0]
    assert payload.view.intensity_1d is arrays[
        browse_1d_row_name("q", "intensity")
    ]
    assert payload.view.axis_2d_x is None
    assert payload.view.axis_2d_y is None
    assert payload.view.intensity_2d is None
    assert payload.view.thumbnail is (
        thumbnail if has_thumbnail else None
    )
    assert payload.view.raw is (
        raw
        if shape in {"one-d-detector", "empty-record-proven-raw"}
        else None
    )
    outcome.borrow_bundle.release()
    assert scope[5].outstanding_borrows == 0
    _close(scope)


@pytest.mark.parametrize(
    "sabotage", ("source", "top-source", "scan", "record-label"),
)
def test_current_heavy_requires_exact_scalar_source_and_record(
    tmp_path, sabotage,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow

    row = FrameScalarRow(
        1,
        source_path="detector.tif",
        source_frame_index=7,
        modes_1d=("q",),
        active_mode_1d="q",
        modes_2d=("map",),
        active_mode_2d="map",
        two_d_kinds=(("map", TwoDKind.Q_CHI),),
    )
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    view = FrameView(
        1,
        axis_2d_x=Axis("Q", "A^-1", False, _readonly([1.0, 2.0])),
        axis_2d_y=Axis("chi", "deg", False, _readonly([3.0, 4.0])),
        intensity_2d=_readonly([[5.0, 6.0], [7.0, 8.0]]),
        source_path=(
            "foreign-detector.tif"
            if sabotage == "top-source" else row.source_path
        ),
        source_frame_index=row.source_frame_index,
    )
    record = FrameRecord(
        2 if sabotage == "record-label" else 1,
        results_2d={"map": view},
        active_mode_2d="map",
    )
    source = canonical_browse_source_identity(row, scope[0].requested_path)
    if sabotage == "source":
        source = "foreign-detector.tif#7"
    scope[6].upsert(FramePublication(
        view,
        record=record,
        source_identity=source,
        scan_key=(
            "foreign-scan" if sabotage == "scan" else scope[0].scan_key
        ),
    ))

    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == () and outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    _close(scope)


@pytest.mark.parametrize(
    "component",
    (
        "axis_2d_x",
        "axis_2d_y",
        "intensity_2d",
        "sigma_2d",
        "raw",
        "thumbnail",
    ),
)
def test_current_heavy_refuses_record_component_identity_drift(
    tmp_path, component,
) -> None:
    from dataclasses import replace

    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow

    row = FrameScalarRow(
        1,
        modes_1d=("q",),
        active_mode_1d="q",
        modes_2d=("map",),
        active_mode_2d="map",
        two_d_kinds=(("map", TwoDKind.Q_CHI),),
    )
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    view = FrameView(
        1,
        axis_2d_x=Axis("Q", "A^-1", False, _readonly([1.0, 2.0])),
        axis_2d_y=Axis("chi", "deg", False, _readonly([3.0, 4.0])),
        intensity_2d=_readonly([[5.0, 6.0], [7.0, 8.0]]),
        sigma_2d=_readonly([[0.5, 0.6], [0.7, 0.8]]),
        raw=_readonly([[9.0, 10.0], [11.0, 12.0]]),
        thumbnail=_readonly([[13.0, 14.0], [15.0, 16.0]]),
    )
    replacements = {
        "axis_2d_x": Axis("Q", "A^-1", False, _readonly([1.0, 2.0])),
        "axis_2d_y": Axis("chi", "deg", False, _readonly([3.0, 4.0])),
        "intensity_2d": _readonly([[5.0, 6.0], [7.0, 8.0]]),
        "sigma_2d": _readonly([[0.5, 0.6], [0.7, 0.8]]),
        "raw": _readonly([[9.0, 10.0], [11.0, 12.0]]),
        "thumbnail": _readonly([[13.0, 14.0], [15.0, 16.0]]),
    }
    record_source = replace(view, **{component: replacements[component]})
    record = FrameRecord(
        1,
        results_2d={"map": record_source},
        active_mode_2d="map",
    )
    scope[6].upsert(FramePublication(
        view,
        record=record,
        source_identity=canonical_browse_source_identity(
            row, scope[0].requested_path,
        ),
        scan_key=scope[0].scan_key,
    ))

    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == () and outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    _close(scope)


def test_current_heavy_refuses_bool_view_and_record_labels(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow

    row = FrameScalarRow(
        1,
        modes_1d=("q",),
        active_mode_1d="q",
        modes_2d=("map",),
        active_mode_2d="map",
        two_d_kinds=(("map", TwoDKind.Q_CHI),),
    )
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    view = FrameView(
        True,
        axis_2d_x=Axis("Q", "A^-1", False, _readonly([1.0, 2.0])),
        axis_2d_y=Axis("chi", "deg", False, _readonly([3.0, 4.0])),
        intensity_2d=_readonly([[5.0, 6.0], [7.0, 8.0]]),
    )
    record = FrameRecord(
        True,
        results_2d={"map": view},
        active_mode_2d="map",
    )
    scope[6].upsert(FramePublication(
        view,
        record=record,
        source_identity=canonical_browse_source_identity(
            row, scope[0].requested_path,
        ),
        scan_key=scope[0].scan_key,
    ))

    # PublicationStore's integer key space aliases True and 1.  Projection
    # admission must still reject the non-exact persisted label types.
    assert scope[6].get(1) is not None
    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == () and outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    _close(scope)


def test_first_preview_hydration_stamps_scalar_catalog_named_modes(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.hydration_transport import (
        PreparedHydrationCommit,
    )
    from xdart.modules.display_context import HydrationRequest
    from xrd_tools.core import Axis, FrameView, TwoDKind
    from xrd_tools.io import FrameScalarRow
    from xrd_tools.io.frame_preview import FramePreview
    from xrd_tools.session.hydration import (
        HydrationOutcome,
        HydrationPurpose,
        HydrationReadKey,
        HydrationScope,
        HydrationToken,
    )

    row = FrameScalarRow(
        1,
        has_thumbnail=True,
        modes_1d=("q_ip",),
        active_mode_1d="q_ip",
        modes_2d=("q_ip_q_oop",),
        active_mode_2d="q_ip_q_oop",
        two_d_kinds=(("q_ip_q_oop", TwoDKind.QIP_QOOP),),
    )
    scope = _scope(
        tmp_path,
        (row,),
        (("q_ip", "Q-ip", "A^-1", False),),
    )
    context = scope[0]
    view = FrameView(
        1,
        axis_1d=Axis("Q-ip", "A^-1", False, _readonly([1.0, 2.0])),
        intensity_1d=_readonly([3.0, 4.0]),
        axis_2d_x=Axis("Q-ip", "A^-1", False, _readonly([1.0, 2.0])),
        axis_2d_y=Axis("Q-oop", "A^-1", False, _readonly([3.0, 4.0])),
        intensity_2d=_readonly([[5.0, 6.0], [7.0, 8.0]]),
        two_d_kind=TwoDKind.QIP_QOOP,
        thumbnail=_readonly([[9.0, 10.0], [11.0, 12.0]]),
    )
    owner = _BrowseHydrationOwner(context)
    hydration_owner = context.hydration_owner
    read_key = HydrationReadKey(
        HydrationScope(*hydration_owner.as_tuple()),
        context.requested_path,
        1,
        HydrationPurpose.PREVIEW,
    )
    token = HydrationToken(read_key, scope[1].display_generation)
    request = HydrationRequest(
        1,
        HydrationPurpose.PREVIEW,
        scope[1].display_generation,
        hydration_owner,
        (scope[6],),
        context.commit_gate,
        read_key=read_key,
        token=token,
    )
    preview = FramePreview(
        read_key,
        view,
        view.thumbnail,
        None,
        None,
        None,
        None,
        None,
    )
    prepared = PreparedHydrationCommit(
        request, token, None, False, preview, None,
    )
    try:
        assert owner.commit_preview(prepared) is HydrationOutcome.HYDRATED
        publication = scope[6].get(1)
        assert publication is not None
        assert publication.record.active_mode_1d == "q_ip"
        assert tuple(publication.record.results_1d) == ("q_ip",)
        assert publication.record.active_mode_2d == "q_ip_q_oop"
        assert tuple(publication.record.results_2d) == ("q_ip_q_oop",)
    finally:
        assert owner.retire()
        _close(scope)


def test_wrong_key_live_borrow_is_adopted_for_exact_cleanup(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xrd_tools.io import Browse1DCache, FrameScalarRow, browse_1d_row_name

    row = FrameScalarRow(1, modes_1d=("q",), active_mode_1d="q")
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    foreign = Browse1DCache(1 << 20)
    name = browse_1d_row_name("q", "axis")
    operation = foreign.begin_store(9, 9, ((name, _readonly([1.0, 2.0])),))
    assert operation.run() == "accepted"
    wrong = foreign.borrow(9, 9, name)
    real_borrow = Browse1DCache.borrow

    def borrow(cache, frame, label, row_name):
        if cache is scope[5]:
            return wrong
        return real_borrow(cache, frame, label, row_name)

    monkeypatch.setattr(Browse1DCache, "borrow", borrow)
    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == () and outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    assert foreign.outstanding_borrows == 0
    _close(scope)
    foreign.close()


def test_multi_frame_eviction_during_acquire_returns_no_partial_payload(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xrd_tools.io import Browse1DCache, FrameScalarRow

    rows = tuple(
        FrameScalarRow(label, modes_1d=("q",), active_mode_1d="q")
        for label in (1, 2)
    )
    scope = _scope(
        tmp_path,
        rows,
        (("q", "Q", "A^-1", False),),
        budget=4 * 3 * np.dtype(np.float64).itemsize,
    )
    for ordinal, label in enumerate((1, 2), 1):
        _seed_label(scope[7], scope[5], scope[4], ordinal, label)
    real_borrow = Browse1DCache.borrow
    calls = []

    def borrow(cache, frame, label, name):
        calls.append((frame, label, name))
        if cache is scope[5] and len(calls) == 3:
            competitor = _readonly(np.arange(6, dtype=np.float64))
            assert cache.begin_store(
                99, 99, (("competitor", competitor),),
            ).run() == "accepted"
        return real_borrow(cache, frame, label, name)

    monkeypatch.setattr(Browse1DCache, "borrow", borrow)
    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.INCOMPLETE
    assert outcome.payloads == ()
    assert outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    assert any(key.name == "competitor" for key in scope[5].resident_keys)
    assert scope[8] == []
    _close(scope)


def test_foreign_generation_context_frame_and_path_are_refused(tmp_path) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
        project_browse_1d,
    )
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xdart.modules.display_context import DisplaySelection
    from xrd_tools.io import FrameScalarRow

    row = FrameScalarRow(1, modes_1d=("q",), active_mode_1d="q")
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    stale = DisplaySelection.for_context(
        scope[0], scope[1].display_generation + 1,
    )
    clone = DisplayFrameKey(
        scope[3][0].run_identity,
        scope[3][0].source_scan,
        scope[3][0].artifact,
        scope[3][0].local_frame_label,
        scope[3][0].work_ordinal,
    )
    wrong_path = DisplayFrameKey(
        scope[3][0].run_identity,
        scope[3][0].source_scan,
        scope[3][0].artifact + ".foreign",
        scope[3][0].local_frame_label,
        scope[3][0].work_ordinal,
    )
    wrong_navigation = FrameNavigationProjection(
        (wrong_path,), wrong_path, (wrong_path,),
    )
    other = _scope(
        tmp_path / "other", (row,), (("q", "Q", "A^-1", False),),
    )
    cases = (
        (scope[0], scope[7], stale, scope[2], scope[3], scope[1]),
        (other[0], scope[7], scope[1], scope[2], scope[3], scope[1]),
        (scope[0], scope[7], scope[1], scope[2], (clone,), scope[1]),
        (
            scope[0], scope[7], scope[1], wrong_navigation,
            (wrong_path,), scope[1],
        ),
    )
    for context, hydration, selection, navigation, targets, current in cases:
        outcome = project_browse_1d(
            context,
            hydration,
            selection,
            navigation,
            targets,
            current_selection=current,
        )
        assert outcome.status is Browse1DProjectionStatus.REFUSED
        assert outcome.payloads == () and outcome.borrow_bundle is None
    assert scope[5].outstanding_borrows == 0
    assert other[5].outstanding_borrows == 0
    _close(scope)
    _close(other)


@pytest.mark.parametrize("cut", ("before", "after"))
def test_borrow_bundle_release_failure_retains_exact_retry_custody(
    tmp_path, monkeypatch, cut,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DBorrowBundle,
        Browse1DProjectionStatus,
    )
    from xrd_tools.io import FrameScalarRow
    from xrd_tools.io.browse_1d_cache import Browse1DBorrow

    row = FrameScalarRow(1, modes_1d=("q",), active_mode_1d="q")
    scope = _scope(tmp_path, (row,), (("q", "Q", "A^-1", False),))
    _seed_label(scope[7], scope[5], scope[4], 1, 1)
    outcome = _project(scope)
    assert outcome.status is Browse1DProjectionStatus.COMPLETE
    bundle = outcome.borrow_bundle
    assert type(bundle) is Browse1DBorrowBundle
    real_release = Browse1DBorrow.release
    calls = []

    def release(borrowed):
        calls.append(borrowed)
        if len(calls) == 1:
            if cut == "after":
                real_release(borrowed)
            raise RuntimeError(f"{cut} release cut")
        return real_release(borrowed)

    monkeypatch.setattr(Browse1DBorrow, "release", release)
    with pytest.raises(RuntimeError, match="release cut"):
        bundle.release()
    assert not bundle.released
    assert bundle.remaining == (2 if cut == "before" else 1)
    bundle.release()
    bundle.release()
    assert bundle.released and bundle.remaining == 0
    assert scope[5].outstanding_borrows == 0
    for alias in (copy, deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            alias(bundle)
    _close(scope)
