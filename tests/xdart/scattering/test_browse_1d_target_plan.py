"""Focused C7b target-planning and dormant cache-runtime oracles."""

from __future__ import annotations

from copy import copy, deepcopy, replace
import pickle

import numpy as np
import pytest


def _readonly(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _rig(tmp_path, count: int, *, norm_channels=None):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.gui.tabs.scattering.context_runtime import _ContextRuntime
    from xdart.modules.display_context import BrowseContext
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.io import Browse1DCache, FrameScalarCatalog, FrameScalarRow
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.session import ScanNormAggregate

    path = tmp_path / f"plan-{count}.nxs"
    path.write_bytes(b"stable scalar-catalog artifact")
    rows = tuple(
        FrameScalarRow(
            label,
            source_path="detector.tif",
            source_frame_index=label - 1,
            modes_1d=("q",),
            active_mode_1d="q",
        )
        for label in range(1, count + 1)
    )
    catalog = FrameScalarCatalog(
        str(path.resolve()),
        "entry",
        rows,
        (("q", "Q", "A^-1", False),),
    )
    cache = Browse1DCache(8 << 20)
    request = BrowseLoadRequest("browse-plan", 1, str(path.resolve()))
    norm_aggregate = (
        None
        if norm_channels is None
        else ScanNormAggregate(
            (request.token, "scan", request.source_path),
            1,
            count,
            norm_channels,
        )
    )
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
        publication_store=PublicationStore(
            max_items=16,
            max_heavy_items=16,
            max_thumbnail_items=16,
        ),
        record_store={},
        norm_aggregate=norm_aggregate,
        scalar_catalog=catalog,
        browse_1d_cache=cache,
        target_entry=catalog.entry,
        loaded_labels=catalog.labels,
        target_snapshot=capture_target_snapshot(path),
    )
    context.adopt_load_request(request)
    context.mark_loaded()
    runtime = _ContextRuntime()
    selection = runtime.adopt_browse(context, request)
    frames = runtime.navigation.frames
    assert runtime.select_navigation(frames[-1], frames)
    return context, runtime, selection, frames, catalog, cache


def _plan(rig, preferences, *, was_active=False, navigation=None,
          selection=None, owned=None):
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        plan_browse_1d_targets,
    )

    context, runtime, selected, _frames, _catalog, _cache = rig
    actual_selection = selected if selection is None else selection
    actual_navigation = runtime.navigation if navigation is None else navigation
    actual_owned = runtime._browse_frame_by_id if owned is None else owned
    return plan_browse_1d_targets(
        context,
        actual_selection,
        actual_navigation,
        preferences,
        actual_owned,
        current_selection=actual_selection,
        current_navigation=actual_navigation,
        was_waterfall_active=was_active,
    )


def _seed(rig, owner, frames) -> None:
    from xrd_tools.core import Axis
    from xrd_tools.io import Frame1DModeRows, Frame1DRows

    context, _runtime, _selection, _all_frames, catalog, cache = rig
    lane = owner._one_d_lane
    for frame in frames:
        label = frame.local_frame_label
        axis = _readonly((0.1, 0.2, 0.3))
        intensity = _readonly((label, label + 1, label + 2))
        rows = Frame1DRows(
            context.requested_path,
            context.target_entry,
            (label,),
            (
                Frame1DModeRows(
                    "q",
                    Axis("Q", "A^-1", values=axis),
                    (label,),
                    (intensity,),
                    None,
                ),
            ),
            "q",
        )
        receipt = cache.begin_store_1d_label(
            catalog,
            rows,
            frame.work_ordinal,
            label,
        )
        if receipt.operation is not None:
            assert receipt.operation.run() == "accepted"
        with lane._lock:
            lane._known_keys[(frame.work_ordinal, label)] = tuple(receipt.keys)


def _close(rig, owner=None) -> None:
    if owner is not None:
        assert owner._one_d_lane.release()
        assert owner.retire()
    rig[-1].close()


def test_overlay_651_is_sampled_before_hydration_with_terminal_position(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DTargetPlanStatus,
        MAX_BROWSE_1D_DISPLAY_TARGETS,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 651)
    navigation_before = rig[1].navigation
    resident_before = rig[-1].resident_keys
    outcome = _plan(rig, ScientificPreferences(plot_mode="Overlay"))

    assert outcome.status is Browse1DTargetPlanStatus.PLANNED
    plan = outcome.plan
    assert plan.waterfall_active and plan.stacked_options_applied
    assert plan.logical_frames is navigation_before.selected
    assert len(plan.display_targets) == MAX_BROWSE_1D_DISPLAY_TARGETS
    assert plan.display_targets[0] is rig[3][0]
    assert plan.display_targets[-1] is rig[3][-1]
    assert plan.logical_positions[0] == 1
    assert plan.logical_positions[-1] == 651
    assert all(
        plan.logical_frames[position - 1] is target
        for target, position in zip(
            plan.display_targets, plan.logical_positions, strict=True,
        )
    )
    assert rig[1].navigation is navigation_before
    assert rig[-1].resident_keys == resident_before
    _close(rig)


def test_explicit_waterfall_filters_once_then_samples_original_positions(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DTargetPlanStatus,
        MAX_BROWSE_1D_DISPLAY_TARGETS,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions

    rig = _rig(tmp_path, 651)
    preferences = ScientificPreferences(
        plot_mode="Waterfall",
        plot_options=ScientificPlotOptions(
            waterfall_start=2,
            waterfall_stop=650,
            waterfall_step=1,
        ),
    )
    outcome = _plan(rig, preferences)

    assert outcome.status is Browse1DTargetPlanStatus.PLANNED
    plan = outcome.plan
    assert plan.logical_frames == rig[3]
    assert len(plan.display_targets) == MAX_BROWSE_1D_DISPLAY_TARGETS
    assert plan.logical_positions[0] == 2
    assert plan.logical_positions[-1] == 650
    assert plan.display_targets[0] is rig[3][1]
    assert plan.display_targets[-1] is rig[3][649]
    assert all(
        plan.logical_frames[position - 1] is target
        for target, position in zip(
            plan.display_targets, plan.logical_positions, strict=True,
        )
    )
    _close(rig)


@pytest.mark.parametrize(
    "mode,count",
    (("Overlay", 4), ("Waterfall", 3)),
)
def test_below_threshold_stacked_modes_slice_exact_selected_order_once(
    tmp_path, mode, count,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DTargetPlanStatus,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions

    rig = _rig(tmp_path, count)
    outcome = _plan(
        rig,
        ScientificPreferences(
            plot_mode=mode,
            plot_options=ScientificPlotOptions(
                waterfall_start=1,
                waterfall_step=2,
            ),
        ),
    )

    assert outcome.status is Browse1DTargetPlanStatus.PLANNED
    plan = outcome.plan
    assert not plan.waterfall_active
    assert plan.stacked_options_applied
    assert plan.logical_frames == rig[3]
    assert plan.logical_positions == (1, 3)
    assert plan.display_targets == (rig[3][0], rig[3][2])
    _close(rig)


@pytest.mark.parametrize(
    "mode,count,was_active,expected_active,expected_count",
    (
        ("Waterfall", 3, False, False, 3),
        ("Waterfall", 4, False, True, 4),
        ("Overlay", 15, False, False, 15),
        ("Overlay", 16, False, True, 16),
        ("Overlay", 8, True, True, 8),
        ("Overlay", 7, True, False, 7),
        ("Single", 15, False, False, 1),
        ("Single", 16, False, False, 1),
        ("Single", 8, True, False, 1),
        ("Average", 16, True, False, 1),
        ("Sum", 16, True, False, 1),
    ),
)
def test_mode_thresholds_do_not_broaden_nonstacked_targets(
    tmp_path, mode, count, was_active, expected_active, expected_count,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DTargetPlanStatus,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, count)
    outcome = _plan(
        rig,
        ScientificPreferences(plot_mode=mode),
        was_active=was_active,
    )

    assert outcome.status is Browse1DTargetPlanStatus.PLANNED
    assert outcome.plan.waterfall_active is expected_active
    assert len(outcome.plan.display_targets) == expected_count
    if expected_count == 1:
        assert outcome.plan.display_targets == (rig[1].navigation.current,)
    _close(rig)


def test_foreign_duplicate_out_of_order_and_malformed_scope_are_refused(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DTargetPlanStatus,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection

    rig = _rig(tmp_path, 3)
    frames = rig[3]
    preferences = ScientificPreferences(plot_mode="Overlay")

    clone = replace(frames[1])
    foreign = FrameNavigationProjection(
        (frames[0], clone, frames[2]), clone, (frames[0], clone),
    )
    duplicate = FrameNavigationProjection(frames, frames[1], frames[:2])
    object.__setattr__(duplicate, "selected", (frames[0], frames[0]))
    out_of_order = FrameNavigationProjection(
        frames, frames[1], (frames[1], frames[0]),
    )
    malformed = replace(preferences, plot_mode="Foreign")

    outcomes = (
        _plan(rig, preferences, navigation=foreign),
        _plan(rig, preferences, navigation=duplicate),
        _plan(rig, preferences, navigation=out_of_order),
        _plan(rig, malformed),
    )
    assert all(
        outcome.status is Browse1DTargetPlanStatus.REFUSED
        and outcome.plan is None
        for outcome in outcomes
    )
    assert rig[-1].resident_keys == ()
    _close(rig)


def test_runtime_incomplete_submits_only_sampled_targets_and_no_ledger(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        MAX_BROWSE_1D_DISPLAY_TARGETS,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 651)
    runtime = rig[1]
    owner = _BrowseHydrationOwner(rig[0])
    calls = []
    submission = object()

    def submit(selection, targets):
        calls.append((selection, targets))
        return submission

    def reader_bomb(*_args, **_kwargs):
        raise AssertionError("runtime attempted an HDF read")

    owner._one_d_lane.submit = submit
    owner._one_d_lane._open_reader = reader_bomb
    committed_scope = object()
    pending = object()
    runtime._committed_trace_scope = committed_scope
    runtime._committed_trace_selection = (rig[3][0],)
    runtime._pending_trace_projection = pending

    outcome = runtime.project_browse_1d_cache(
        owner,
        preferences=ScientificPreferences(plot_mode="Overlay"),
        was_waterfall_active=False,
    )

    assert outcome.status is Browse1DProjectionStatus.INCOMPLETE
    assert outcome.submission_identity is submission
    assert len(calls) == 1
    assert calls[0][0] is rig[2]
    assert calls[0][1] is outcome.plan.display_targets
    assert len(calls[0][1]) == MAX_BROWSE_1D_DISPLAY_TARGETS
    assert calls[0][1][0] is rig[3][0]
    assert calls[0][1][-1] is rig[3][-1]
    assert runtime._committed_trace_scope is committed_scope
    assert runtime._committed_trace_selection == (rig[3][0],)
    assert runtime._pending_trace_projection is pending
    _close(rig, owner)


def test_runtime_complete_cache_retains_linear_bundle_and_performs_no_read(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 3)
    owner = _BrowseHydrationOwner(rig[0])
    target = rig[1].navigation.current
    _seed(rig, owner, (target,))
    reads = []

    def reader_bomb(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("complete cache attempted an HDF read")

    owner._one_d_lane._open_reader = reader_bomb
    owner._one_d_lane.submit = lambda *_args: pytest.fail(
        "complete cache submitted hydration"
    )
    outcome = rig[1].project_browse_1d_cache(
        owner,
        preferences=ScientificPreferences(plot_mode="Single"),
        was_waterfall_active=False,
    )

    assert outcome.status is Browse1DProjectionStatus.COMPLETE
    assert outcome.payloads[0].frame_key is target
    assert outcome.borrow_bundle is not None
    assert not outcome.borrow_bundle.released
    assert outcome.borrow_bundle.remaining == 2
    assert reads == []
    for operation in (copy, deepcopy):
        with pytest.raises(TypeError):
            operation(outcome)
    with pytest.raises(TypeError):
        pickle.dumps(outcome)
    outcome.borrow_bundle.release()
    assert outcome.borrow_bundle.released
    _close(rig, owner)


def test_runtime_refused_scope_has_no_submission_or_cache_mutation(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.modules.display_context import DisplaySelection, HydrationOwner

    rig = _rig(tmp_path, 3)
    owner = _BrowseHydrationOwner(rig[0])
    calls = []
    owner._one_d_lane.submit = lambda *_args: calls.append(True)
    before = rig[-1].resident_keys
    selected = rig[2]
    rig[1]._selection = DisplaySelection(
        selected.kind,
        HydrationOwner(
            selected.context_token,
            selected.scan_key,
            "foreign-artifact",
            selected.owner.epoch,
        ),
        selected.display_generation,
    )

    outcome = rig[1].project_browse_1d_cache(
        owner,
        preferences=ScientificPreferences(plot_mode="Overlay"),
        was_waterfall_active=False,
    )

    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == () and outcome.borrow_bundle is None
    assert calls == []
    assert rig[-1].resident_keys == before
    _close(rig, owner)


def test_stale_complete_is_refused_without_dropping_borrow_custody(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.modules.display_context import DisplaySelection

    rig = _rig(tmp_path, 3)
    runtime = rig[1]
    owner = _BrowseHydrationOwner(rig[0])
    target = runtime.navigation.current
    _seed(rig, owner, (target,))
    original = owner.project_1d

    def drift(*args, **kwargs):
        projected = original(*args, **kwargs)
        selected = runtime._selection
        runtime._selection = DisplaySelection(
            selected.kind,
            selected.owner,
            selected.display_generation + 1,
        )
        return projected

    owner.project_1d = drift
    outcome = runtime.project_browse_1d_cache(
        owner,
        preferences=ScientificPreferences(plot_mode="Single"),
        was_waterfall_active=False,
    )

    assert outcome.status is Browse1DProjectionStatus.REFUSED
    assert outcome.payloads == ()
    assert outcome.borrow_bundle is not None
    assert not outcome.borrow_bundle.released
    assert outcome.borrow_bundle.remaining == 2
    outcome.borrow_bundle.release()
    _close(rig, owner)
