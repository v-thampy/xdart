"""Focused failure-atomic adoption oracles for cache-backed Browse 1-D."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_browse_1d_target_plan import (
    _close,
    _plan,
    _rig,
    _seed,
)


def test_complete_651_copy_releases_before_qt_and_keeps_logical_extent(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_display import (
        MAX_BROWSE_1D_DETACHED_ROOTS,
        prepare_browse_1d_display,
    )
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        MAX_BROWSE_1D_DISPLAY_TARGETS,
    )
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xdart.gui.tabs.scattering.shell_values import (
        AxisProjection,
        ScientificProjection,
        TraceProjection,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rig = _rig(tmp_path, 651)
    owner = _BrowseHydrationOwner(rig[0])
    # The exact current frame is authoritative even when an ordinary bounded
    # sample would omit it; it remains one of the copied display rows.
    assert rig[1].select_navigation(rig[3][1], rig[3])
    preferences = ScientificPreferences(plot_mode="Overlay")
    planned = _plan(rig, preferences).plan
    assert planned is not None
    assert rig[3][1] in planned.display_targets
    _seed(rig, owner, planned.display_targets)
    runtime = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    assert runtime.status is Browse1DProjectionStatus.COMPLETE
    # Exercise the former 3 * 256 worst case: every sampled payload carries
    # three distinct exact sources, but detached storage still owns one packed
    # root and retains all 256 display rows.
    runtime = replace(
        runtime,
        payloads=tuple(
            replace(
                payload,
                view=replace(
                    payload.view,
                    sigma_1d=payload.view.intensity_1d.copy(),
                ),
            )
            for payload in runtime.payloads
        ),
    )
    source_arrays = tuple(
        array
        for payload in runtime.payloads
        for array in (
            payload.view.axis_1d.values,
            payload.view.intensity_1d,
            payload.view.sigma_1d,
        )
    )
    assert len({id(array) for array in source_arrays}) == (
        3 * MAX_BROWSE_1D_DISPLAY_TARGETS
    )
    bundle = runtime.borrow_bundle
    assert bundle is not None and not bundle.released

    detached = prepare_browse_1d_display(runtime)
    assert detached is not None
    assert bundle.released
    assert len(detached.payloads) == MAX_BROWSE_1D_DISPLAY_TARGETS
    assert len(detached.trace_snapshot.logical_frames) == 651
    assert detached.trace_snapshot.logical_positions[-1] == 651
    copied_arrays = tuple(
        array
        for payload in detached.payloads
        for array in (
            payload.view.axis_1d.values,
            payload.view.intensity_1d,
            payload.view.sigma_1d,
        )
    )
    copied_roots = {id(array.base): array.base for array in copied_arrays}
    assert len(copied_roots) == 1
    assert len(copied_roots) <= MAX_BROWSE_1D_DETACHED_ROOTS
    assert all(
        type(root) is np.ndarray
        and root.base is None
        and not root.flags.writeable
        for root in copied_roots.values()
    )
    assert all(not array.flags.writeable for array in copied_arrays)

    traces = tuple(
        TraceProjection(
            payload.frame_key,
            AxisProjection(
                payload.view.axis_1d.values,
                payload.view.axis_1d.label,
                payload.view.axis_1d.unit,
            ),
            payload.view.intensity_1d,
            payload.title,
        )
        for payload in detached.payloads
    )
    base = ScientificProjection(
        traces=traces,
        processing_mode="Int 2D",
        plot_mode="Overlay",
        plot_options=preferences.plot_options,
        # Sparse 1-D adoption must not wait for the independent exact-current
        # detector/cake preview to become resident.
        retain_display=True,
    )
    snapshot = replace(
        detached.trace_snapshot,
        science_contract=base.browse_science_contract,
    )
    scientific = replace(base, browse_trace_snapshot=snapshot)
    view = ScientificView()
    try:
        prior = make_shell_projection(
            frame_count=1,
            heavy_indices=(0,),
            source_scan="prior-artifact",
        )
        view.reconcile(
            prior.scientific,
            prior.navigation,
            completed=1,
            total=1,
            detail="Prior",
        )
        assert view.raw.canvas.displayed_image is not None
        assert view.cake.canvas.displayed_image is not None
        # Qt receives only the detached state; no borrow-bearing object exists
        # in this call or in the accepted ScientificProjection.
        view.reconcile(
            scientific,
            rig[1].navigation,
            completed=651,
            total=651,
            detail="Ready",
        )
        assert view.trace_row_count == 651
        assert view.trace_history_keys == rig[1].navigation.selected
        assert len(view._waterfall_source_keys) == MAX_BROWSE_1D_DISPLAY_TARGETS
        assert view._waterfall_y_values[0] == 1.0
        assert view._waterfall_y_values[-1] == 651.0
        # The sparse trace receipt is authoritative without a heavy preview;
        # prior-artifact detector/cake pixels must not survive under its new
        # title and trace state.
        assert view.raw.canvas.displayed_image.size == 0
        assert view.cake.canvas.displayed_image.size == 0
    finally:
        view.close()
        _close(rig, owner)


@pytest.mark.parametrize("status", ("incomplete", "refused"))
def test_incomplete_and_refused_are_preserve_current_outcomes(
    tmp_path, status,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_display import (
        Browse1DDisplayRefusal,
        prepare_browse_1d_display,
    )
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DRuntimeProjection,
    )
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 4)
    plan = _plan(rig, ScientificPreferences(plot_mode="Overlay")).plan
    assert plan is not None
    runtime = (
        Browse1DRuntimeProjection(
            Browse1DProjectionStatus.INCOMPLETE,
            plan=plan,
            submission_identity=object(),
        )
        if status == "incomplete"
        else Browse1DRuntimeProjection(
            Browse1DProjectionStatus.REFUSED,
            plan=plan,
            diagnostic="preserve",
        )
    )
    try:
        if status == "incomplete":
            assert prepare_browse_1d_display(runtime) is None
        else:
            with pytest.raises(Browse1DDisplayRefusal, match="^preserve$"):
                prepare_browse_1d_display(runtime)
    finally:
        _close(rig)


def test_page_fail_closes_terminal_and_incompatible_browse_cache_misses(
    tmp_path, monkeypatch,
) -> None:
    import time

    from tests.xdart.scattering.test_e4_preview_transport import (
        _write_processed,
    )
    from tests.xdart.scattering.test_p3_experiment_operation_composition import (
        _close as _close_page,
        _page,
    )
    from xdart.gui.tabs.scattering.browse_1d_display import (
        prepare_browse_1d_display,
    )
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.browse_1d_target_plan import (
        Browse1DRuntimeProjection,
    )
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xdart.gui.tabs.scattering.shell_values import BrowseTraceSnapshot

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processed, _raw = _write_processed(
        tmp_path / "legacy",
        labels=tuple(range(1, 17)),
    )
    page, _store = _page(tmp_path, monkeypatch)
    controller = page._context_controller

    def wait_for(call):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            value = call()
            if value is not None:
                return value
            time.sleep(0.005)
        raise AssertionError("timed out waiting for focused Browse setup")

    def mount_prior_waterfall(*, include_snapshot=True, template=None):
        prior = make_shell_projection(
            frame_count=8,
            selected_index=7,
            heavy_indices=(7,),
            plot_mode="Waterfall",
            source_scan="eiger-prior",
        )
        base = (
            replace(
                prior.scientific,
                processing_mode="Int 2D",
                norm_channels=(page._preferences.norm_channel,),
                norm_channel=page._preferences.norm_channel,
                color_map=page._preferences.color_map,
                plot_axis=page._preferences.plot_axis,
                plot_mode="Waterfall",
                share_axis=page._preferences.share_axis,
                plot_options=page._preferences.plot_options,
            )
            if template is None
            else replace(
                template,
                heavy_available=prior.scientific.heavy_available,
                traces=prior.scientific.traces,
                heavy=prior.scientific.heavy,
                title=prior.scientific.title,
                browse_trace_snapshot=None,
            )
        )
        snapshot = BrowseTraceSnapshot(
            logical_frames=prior.navigation.selected,
            display_frames=tuple(trace.frame for trace in base.traces),
            logical_positions=tuple(range(1, len(base.traces) + 1)),
            plot_mode="Waterfall",
            waterfall_active=True,
            stacked_options_applied=True,
            science_contract=base.browse_science_contract,
        )
        scientific = (
            replace(base, browse_trace_snapshot=snapshot)
            if include_snapshot
            else base
        )
        page._shell.scientific.reconcile(
            scientific,
            prior.navigation,
            completed=8,
            total=8,
            detail="Prior Eiger",
        )
        page._last_scientific_projection = scientific
        assert page._shell.scientific.bottom_waterfall_active
        assert page._shell.scientific.trace_row_count == 8
        assert page._shell.scientific.raw.canvas.displayed_image.size
        return scientific

    try:
        request = controller.begin_browse(str(processed.resolve()))
        outcome = wait_for(controller.poll_browse)
        assert outcome.request is request
        assert outcome.status is BrowseLoadStatus.READY
        navigation = controller.navigation
        assert len(navigation.frames) == 16
        assert controller.select_navigation(
            navigation.frames[-1], navigation.frames,
        )
        page._preferences = replace(
            page._preferences, plot_mode="Waterfall",
        )
        view = page._shell.scientific
        original_project = controller.project_browse_1d_cache

        def refresh_current_cache():
            page._refresh_shell()
            candidate = page._last_scientific_projection
            snapshot = (
                None if candidate is None
                else candidate.browse_trace_snapshot
            )
            return (
                candidate
                if snapshot is not None
                and all(
                    frame.artifact == str(processed.resolve())
                    for frame in snapshot.logical_frames
                )
                else None
            )

        exact_current = wait_for(refresh_current_cache)
        exact_snapshot = exact_current.browse_trace_snapshot
        assert exact_snapshot is not None
        exact_history = view.trace_history_keys
        assert exact_history == exact_snapshot.logical_frames

        controller.capture_norm_aggregate_for_refresh()
        same_contract_planned = original_project(
            preferences=page._preferences,
            was_waterfall_active=True,
        )
        assert same_contract_planned.status in {
            Browse1DProjectionStatus.COMPLETE,
            Browse1DProjectionStatus.INCOMPLETE,
        }
        if same_contract_planned.status is Browse1DProjectionStatus.COMPLETE:
            same_contract_detached = prepare_browse_1d_display(
                same_contract_planned,
            )
            assert same_contract_detached is not None
            same_contract_plan = same_contract_detached.plan
        else:
            same_contract_plan = same_contract_planned.plan
        assert same_contract_plan is not None
        same_contract_incomplete = Browse1DRuntimeProjection(
            Browse1DProjectionStatus.INCOMPLETE,
            plan=same_contract_plan,
            submission_identity=object(),
        )
        monkeypatch.setattr(
            controller,
            "project_browse_1d_cache",
            lambda **_kwargs: same_contract_incomplete,
        )

        # One exact-current copied snapshot remains visible while its own
        # unchanged sparse cache retry is incomplete.
        page._refresh_shell()
        assert view.trace_history_keys == exact_history
        assert view.bottom_waterfall_active
        assert page._scientific_repaint_pending

        # A matching presentation contract cannot authorize an old artifact's
        # copied Browse rows.  Exact current request/navigation/frame custody
        # is required before an INCOMPLETE retry may retain prior science.
        foreign = mount_prior_waterfall(template=exact_current)
        assert (
            foreign.browse_trace_snapshot.science_contract
            == exact_snapshot.science_contract
        )
        page._refresh_shell()
        assert view.trace_history_keys == ()
        assert not view.bottom_waterfall_active
        assert view.raw.canvas.displayed_image.size == 0
        assert view.cake.canvas.displayed_image.size == 0
        assert page._scientific_repaint_pending

        # Absence of a BrowseTraceSnapshot is never evidence that arbitrary
        # outgoing science belongs to the currently adopted Browse artifact.
        foreign = mount_prior_waterfall(
            include_snapshot=False,
            template=exact_current,
        )
        assert (
            foreign.browse_science_contract
            == exact_snapshot.science_contract
        )
        page._refresh_shell()
        assert view.trace_history_keys == ()
        assert not view.bottom_waterfall_active
        assert view.raw.canvas.displayed_image.size == 0
        assert view.cake.canvas.displayed_image.size == 0
        assert page._scientific_repaint_pending

        monkeypatch.setattr(
            controller, "project_browse_1d_cache", original_project,
        )
        mount_prior_waterfall()
        refused = Browse1DRuntimeProjection(
            Browse1DProjectionStatus.REFUSED,
            diagnostic="terminal old-artifact cache refusal",
        )
        monkeypatch.setattr(
            controller,
            "project_browse_1d_cache",
            lambda **_kwargs: refused,
        )
        page._refresh_shell()
        assert page._shell.scientific is view
        assert view.trace_history_keys == ()
        assert not view.bottom_waterfall_active
        assert view.raw.canvas.displayed_image.size == 0
        assert view.cake.canvas.displayed_image.size == 0
        assert view.title.text() == "Current"
        assert page._notice_text == "terminal old-artifact cache refusal"
        assert not page._scientific_repaint_pending

        monkeypatch.setattr(
            controller, "project_browse_1d_cache", original_project,
        )
        mount_prior_waterfall()
        page._preferences = replace(page._preferences, plot_mode="Single")
        controller.capture_norm_aggregate_for_refresh()
        planned = original_project(
            preferences=page._preferences,
            was_waterfall_active=True,
        )
        assert planned.status in {
            Browse1DProjectionStatus.COMPLETE,
            Browse1DProjectionStatus.INCOMPLETE,
        }
        if planned.status is Browse1DProjectionStatus.COMPLETE:
            detached = prepare_browse_1d_display(planned)
            assert detached is not None
            plan = detached.plan
        else:
            plan = planned.plan
        assert plan is not None
        incomplete = Browse1DRuntimeProjection(
            Browse1DProjectionStatus.INCOMPLETE,
            plan=plan,
            submission_identity=object(),
        )
        monkeypatch.setattr(
            controller,
            "project_browse_1d_cache",
            lambda **_kwargs: incomplete,
        )
        page._refresh_shell()
        assert view.presentation_plot_mode == "Single"
        assert view.trace_history_keys == ()
        assert not view.bottom_waterfall_active
        assert view.title.text() == "Current"
        assert page._scientific_repaint_pending

        mount_prior_waterfall()
        page._preferences = replace(page._preferences, plot_mode="Average")
        page._refresh_shell()
        assert view.presentation_plot_mode == "Average"
        assert view.trace_history_keys == ()
        assert not view.bottom_waterfall_active
        assert not page._scientific_repaint_pending
        assert page._notice_text.startswith(
            "Browse cache display is unavailable for Average/Sum"
        )
    finally:
        _close_page(page, app)


@pytest.mark.parametrize("two_d", [True, False])
def test_single_browse_chi_slice_uses_saved_cake_and_keeps_images(
    tmp_path, monkeypatch, two_d,
) -> None:
    import time
    from tests.xdart.scattering.test_e4_preview_transport import _write_processed
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _close, _page
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processed, _ = _write_processed(tmp_path / "slice", schema_version=3, two_d=two_d)
    page, _ = _page(tmp_path, monkeypatch)
    controller = page._context_controller

    def wait_for(call):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            app.processEvents()
            value = call()
            if value:
                return value
            time.sleep(0.005)
        raise AssertionError(("Browse slice timed out", page._notice_text))

    try:
        request = controller.begin_browse(str(processed.resolve()))
        loaded = wait_for(controller.poll_browse)
        assert loaded.request is request and loaded.status is BrowseLoadStatus.READY
        current = controller.navigation.frames[1]
        assert controller.select_navigation(current, (current,))
        page._refresh_shell()
        view = page._shell.scientific
        # Real controls must request a cake slice, not the stored full 1-D row.
        view.slice_center.setValue(-1.0)
        view.slice_width.setValue(0.1)
        view.slice.click()
        assert page._preferences.slice_enabled

        if not two_d:
            wait_for(lambda: (page._refresh_shell() or
                "no saved 2-D data" in page._notice_text))
            assert not page._last_scientific_projection.traces
            assert not page._scientific_repaint_pending
            return

        def sliced():
            page._refresh_shell()
            projection = page._last_scientific_projection
            return projection if (projection is not None and projection.traces
                and "@" in projection.traces[0].title) else None

        projection = wait_for(sliced)
        np.testing.assert_allclose(projection.traces[0].intensity, [2.0, 4.0, 6.0])
        assert view.raw.canvas.displayed_image.size
        assert view.cake.canvas.displayed_image.size
        assert len(view._slice_extent_lines) == 2
        view.slice_center.setValue(1.0)
        projection = wait_for(lambda: (p if (p := sliced()) is not None
            and np.allclose(p.traces[0].intensity, [3.0, 5.0, 7.0]) else None))
        np.testing.assert_allclose(projection.traces[0].axis.values, [0.1, 0.2, 0.3])
        # A cold selected frame must replace both images and the cut, not keep
        # the preceding frame under the new selection.
        current = controller.navigation.frames[0]
        assert controller.select_navigation(current, (current,))
        projection = wait_for(lambda: (p if (p := sliced()) is not None
            and p.traces[0].frame is current else None))
        np.testing.assert_allclose(projection.traces[0].intensity, [2.0, 4.0, 6.0])
        view.slice.click()
        assert not page._preferences.slice_enabled
        wait_for(lambda: (page._refresh_shell() or
            (page._last_scientific_projection.browse_trace_snapshot is not None)))
        np.testing.assert_allclose(page._last_scientific_projection.traces[0].intensity, [2, 3, 4])
    finally:
        _close(page, app)


def test_cache_paint_exception_replaces_whole_scientific_view(
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.shell_values import (
        BrowseTraceSnapshot,
        ShellCommand,
        ShellCommandKind,
    )
    from xdart.gui.tabs.scattering.workspace_shell import (
        ScatteringWorkspaceShell,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    projection = make_shell_projection(
        frame_count=4,
        plot_mode="Overlay",
        source_scan="scan",
    )
    frames = projection.navigation.selected
    snapshot = BrowseTraceSnapshot(
        frames,
        frames,
        (1, 2, 3, 4),
        plot_mode="Overlay",
        stacked_options_applied=True,
        science_contract=projection.scientific.browse_science_contract,
    )
    projection = replace(
        projection,
        scientific=replace(
            projection.scientific,
            browse_trace_snapshot=snapshot,
        ),
    )
    shell = ScatteringWorkspaceShell()
    old = shell.scientific
    forwarded = []
    shell.commandRequested.connect(forwarded.append)

    def paint_bomb(*_args, **_kwargs):
        raise RuntimeError("paint bomb")

    monkeypatch.setattr(old, "reconcile", paint_bomb)
    try:
        with pytest.raises(RuntimeError, match="paint bomb"):
            shell.apply_state(
                projection,
                replace_scientific_on_failure=True,
            )
        fresh = shell.scientific
        assert fresh is not old
        assert fresh.parent() is shell.splitter
        assert old.parent() is None
        command = ShellCommand(ShellCommandKind.CLEAR_1D)
        fresh.commandRequested.emit(command)
        assert forwarded == [command]
    finally:
        shell.close()


def test_target_plan_requalification_rejects_navigation_replacement(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 8)
    owner = _BrowseHydrationOwner(rig[0])
    plan = _plan(rig, ScientificPreferences(plot_mode="Overlay")).plan
    assert plan is not None
    assert rig[1].browse_1d_plan_is_current(owner, plan)
    assert rig[1].select_navigation(rig[3][0], rig[3])
    assert not rig[1].browse_1d_plan_is_current(owner, plan)
    _close(rig, owner)


def test_cache_single_receipt_reuses_compatible_mounted_item(
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xdart.gui.tabs.scattering.shell_values import (
        BrowseTraceSnapshot,
        FrameNavigationProjection,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    projection = make_shell_projection(
        frame_count=2,
        selected_index=0,
        plot_mode="Single",
        source_scan="scan",
    )
    frames = projection.navigation.frames
    base = replace(
        projection.scientific,
        heavy=None,
        heavy_available=frozenset(),
        plot_mode="Single",
    )

    def state_for(index):
        frame = frames[index]
        state = replace(base, traces=(base.traces[index],))
        snapshot = BrowseTraceSnapshot(
            (frame,),
            (frame,),
            (1,),
            plot_mode="Single",
            science_contract=state.browse_science_contract,
        )
        return (
            replace(state, browse_trace_snapshot=snapshot),
            FrameNavigationProjection(frames, frame, (frame,)),
        )

    view = ScientificView()
    try:
        first, first_navigation = state_for(0)
        view.reconcile(
            first, first_navigation, completed=1, total=2, detail="Ready",
        )
        item = tuple(view.curve.listDataItems())[0]
        original = item.setData
        calls = []

        def capture(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(item, "setData", capture)
        second, second_navigation = state_for(1)
        view.reconcile(
            second, second_navigation, completed=2, total=2, detail="Ready",
        )
        assert tuple(view.curve.listDataItems()) == (item,)
        assert len(calls) == 1
        assert view.trace_history_keys == (frames[1],)
    finally:
        view.close()


def test_detached_copy_dedupes_exact_roots_and_refuses_before_budget_allocation(
    tmp_path, monkeypatch,
) -> None:
    import xdart.gui.tabs.scattering.browse_1d_display as display
    from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xrd_tools.core import Axis

    rig = _rig(tmp_path, 3)
    owner = _BrowseHydrationOwner(rig[0])
    preferences = ScientificPreferences(plot_mode="Overlay")
    plan = _plan(rig, preferences).plan
    assert plan is not None
    _seed(rig, owner, plan.display_targets)
    runtime = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    shared_axis = runtime.payloads[0].view.axis_1d.values
    payloads = tuple(
        replace(
            payload,
            view=replace(
                payload.view,
                axis_1d=Axis(
                    payload.view.axis_1d.label,
                    payload.view.axis_1d.unit,
                    payload.view.axis_1d.log,
                    shared_axis,
                ),
            ),
        )
        for payload in runtime.payloads
    )
    detached = display.prepare_browse_1d_display(
        replace(runtime, payloads=payloads)
    )
    assert detached is not None
    copied_axis = detached.payloads[0].view.axis_1d.values
    assert copied_axis is not shared_axis
    assert all(
        payload.view.axis_1d.values is copied_axis
        for payload in detached.payloads
    )

    over_budget = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    bundle = over_budget.borrow_bundle
    assert bundle is not None and not bundle.released
    with monkeypatch.context() as patcher:
        patcher.setattr(display, "MAX_BROWSE_1D_DETACHED_BYTES", 1)

        def allocation_bomb(_sources):
            pytest.fail("copy allocation ran before budget refusal")

        patcher.setattr(display, "_packed_display_roots", allocation_bomb)
        with pytest.raises(
            display.Browse1DDisplayRefusal,
            match="budget exceeded",
        ):
            display.prepare_browse_1d_display(over_budget)
    assert bundle.released

    over_roots = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    root_bundle = over_roots.borrow_bundle
    assert root_bundle is not None and not root_bundle.released
    with monkeypatch.context() as patcher:
        patcher.setattr(display, "MAX_BROWSE_1D_DETACHED_ROOTS", 0)
        patcher.setattr(display, "_packed_display_roots", allocation_bomb)
        with pytest.raises(
            display.Browse1DDisplayRefusal,
            match="budget exceeded",
        ):
            display.prepare_browse_1d_display(over_roots)
    assert root_bundle.released
    _close(rig, owner)


def test_post_release_snapshot_invariant_is_a_display_refusal(
    tmp_path, monkeypatch,
) -> None:
    import xdart.gui.tabs.scattering.browse_1d_display as display
    from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences

    rig = _rig(tmp_path, 1)
    owner = _BrowseHydrationOwner(rig[0])
    preferences = ScientificPreferences(plot_mode="Single")
    plan = _plan(rig, preferences).plan
    assert plan is not None
    _seed(rig, owner, plan.display_targets)
    runtime = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    bundle = runtime.borrow_bundle
    assert bundle is not None and not bundle.released

    def invariant_bomb(*_args, **_kwargs):
        raise ValueError("snapshot invariant")

    monkeypatch.setattr(display, "BrowseTraceSnapshot", invariant_bomb)
    with pytest.raises(
        display.Browse1DDisplayRefusal,
        match="snapshot invariant",
    ):
        display.prepare_browse_1d_display(runtime)
    assert bundle.released
    _close(rig, owner)


def test_cache_refresh_captures_exact_browse_norm_over_foreign_hold(
    tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_display import (
        prepare_browse_1d_display,
    )
    from xdart.gui.tabs.scattering.browse_hydration import _BrowseHydrationOwner
    from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
    from xrd_tools.session import ScanNormAggregate

    rig = _rig(tmp_path, 2, norm_channels={"i0": (4.0, 2)})
    owner = _BrowseHydrationOwner(rig[0])
    preferences = ScientificPreferences(plot_mode="Overlay")
    plan = _plan(rig, preferences).plan
    assert plan is not None
    _seed(rig, owner, plan.display_targets)
    rig[1]._norm_aggregate = ScanNormAggregate(
        ("foreign", "scan", "/foreign"), 9, 2, {"i0": (8.0, 2)},
    )
    rig[1].capture_norm_aggregate_for_refresh()
    runtime = rig[1].project_browse_1d_cache(
        owner,
        preferences=preferences,
        was_waterfall_active=False,
    )
    assert rig[1].norm_aggregate is rig[0].norm_aggregate
    assert prepare_browse_1d_display(runtime) is not None
    _close(rig, owner)


@pytest.mark.parametrize(
    "status,transient,expected",
    (
        ("incomplete", False, True),
        ("refused", False, False),
        ("complete", False, False),
        ("refused", True, True),
    ),
)
def test_cache_poll_policy_distinguishes_terminal_refusal(
    status, transient, expected,
) -> None:
    from xdart.gui.tabs.scattering.browse_1d_projection import (
        Browse1DProjectionStatus,
    )
    from xdart.gui.tabs.scattering.page import (
        _browse_1d_cache_retry_needed,
    )

    assert _browse_1d_cache_retry_needed(
        Browse1DProjectionStatus(status),
        transient=transient,
    ) is expected


def test_drain_gate_settles_page_debt_before_retry_or_context_mutation(
) -> None:
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace

    class Page:
        def __init__(self):
            self.releases = [False, True]
            self._browse_1d_release_debt = object()
            self.average_retries = 0

        def _release_browse_1d_debt(self):
            released = self.releases.pop(0)
            if released:
                self._browse_1d_release_debt = None
            return released

        def _retry_pending_average_reload(self):
            self.average_retries += 1

    page = Page()
    settle = ScatteringWorkspace._settle_browse_1d_before_drain
    assert not settle(page)
    assert page.average_retries == 0
    assert settle(page)
    assert page.average_retries == 1
    assert page._browse_1d_release_debt is None
