from __future__ import annotations

import gc
from dataclasses import replace
import weakref

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering import scientific_view as scientific_view_module
from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
from xdart.gui.tabs.scattering.state_machine import RunPhase

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_e3_context_contract import _running_controller, _view


def _reconcile(
    view: ScientificView,
    scientific,
    navigation: FrameNavigationProjection,
    *,
    completed: int,
    total: int,
) -> None:
    view.reconcile(
        scientific,
        navigation,
        completed=completed,
        total=total,
        detail="Processing" if completed < total else "Ready",
    )


def _readonly(values, *, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _prefix_state(shell, traces, count: int, **changes):
    frames = shell.navigation.frames
    navigation = FrameNavigationProjection(
        frames,
        frames[count - 1],
        frames[:count],
    )
    scientific = replace(
        shell.scientific,
        traces=traces[:count],
        heavy=replace(shell.scientific.heavy, frame=frames[count - 1]),
        **changes,
    )
    return navigation, scientific


def _expected_aggregate(traces, mode: str, scale: str = "Linear"):
    scaled = tuple(
        replace(
            trace,
            intensity=scientific_view_module._scaled_intensity(
                trace.intensity,
                scale,
            ),
        )
        for trace in traces
    )
    return scientific_view_module.aggregate_traces(scaled, mode)[0].intensity


def test_3621_overlay_keeps_exact_history_but_paints_at_most_256_rows() -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=3621,
        selected_index=3620,
        heavy_indices=(0, 3620),
        plot_mode="Overlay",
    )
    view = ScientificView()
    try:
        _reconcile(
            view,
            shell.scientific,
            shell.navigation,
            completed=3621,
            total=3621,
        )

        assert view._trace_history_keys == shell.navigation.selected
        assert len(view._trace_history_keys) == 3621
        assert len(view.waterfall_source_frame_keys) == 3621
        assert all(
            painted is expected
            for painted, expected in zip(
                view.waterfall_source_frame_keys,
                shell.navigation.selected,
                strict=True,
            )
        )
        source_keys = view._waterfall_source_keys
        foreign = object()
        view._waterfall_source_keys = (("live", id(foreign)),)
        assert view.waterfall_source_frame_keys == ()
        view._waterfall_source_keys = (
            ("pin", id(shell.navigation.selected[0])),
        )
        assert view.waterfall_source_frame_keys == ()
        view._waterfall_source_keys = source_keys
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.waterfall.image.image.shape[1] <= 256
        assert len(view._waterfall_y_values) <= 256
        assert view._waterfall_y_values[0] == 1.0
        assert view._waterfall_y_values[-1] == 3621.0
    finally:
        view.close()


def test_live_waterfall_throttles_before_transform_but_terminal_catches_up(
    monkeypatch,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    clock = {"now": 100.0}
    monkeypatch.setattr(
        scientific_view_module.time,
        "monotonic",
        lambda: clock["now"],
    )
    shell = make_shell_projection(
        frame_count=17,
        selected_index=16,
        heavy_indices=(15, 16),
        plot_mode="Overlay",
    )
    frames = shell.navigation.frames
    first_navigation = FrameNavigationProjection(
        frames,
        frames[15],
        frames[:16],
    )
    first_scientific = replace(
        shell.scientific,
        traces=shell.scientific.traces[:16],
        heavy=replace(shell.scientific.heavy, frame=frames[15]),
        live_update=True,
    )
    delta_scientific = replace(
        shell.scientific,
        traces=(shell.scientific.traces[-1],),
        live_update=True,
    )
    view = ScientificView()
    try:
        _reconcile(
            view,
            first_scientific,
            first_navigation,
            completed=16,
            total=20,
        )
        assert view.waterfall.image.image.shape == (64, 16)
        assert all(
            painted is expected
            for painted, expected in zip(
                view.waterfall_source_frame_keys,
                frames[:16],
                strict=True,
            )
        )

        calls = {"scale": 0, "stack": 0}
        original_scale = scientific_view_module._scaled_intensity
        original_stack = scientific_view_module._waterfall_rows_on_reference_axis

        def counted_scale(values, scale):
            calls["scale"] += 1
            return original_scale(values, scale)

        def counted_stack(traces):
            calls["stack"] += 1
            return original_stack(traces)

        monkeypatch.setattr(
            scientific_view_module,
            "_scaled_intensity",
            counted_scale,
        )
        monkeypatch.setattr(
            scientific_view_module,
            "_waterfall_rows_on_reference_axis",
            counted_stack,
        )

        clock["now"] = 100.1
        _reconcile(
            view,
            delta_scientific,
            shell.navigation,
            completed=17,
            total=20,
        )

        assert calls == {"scale": 0, "stack": 0}
        assert view._trace_history_keys == shell.navigation.selected
        assert len(view.waterfall_source_frame_keys) == 16
        assert all(
            painted is expected
            for painted, expected in zip(
                view.waterfall_source_frame_keys,
                frames[:16],
                strict=True,
            )
        )
        assert view.waterfall.image.image.shape == (64, 16)

        clock["now"] = 100.6
        _reconcile(
            view,
            replace(delta_scientific, live_update=False),
            shell.navigation,
            completed=17,
            total=20,
        )

        assert calls["scale"] == 17
        assert calls["stack"] == 1
        assert view.waterfall.image.image.shape == (64, 17)
        assert view._waterfall_y_values[-1] == 17.0
        assert len(view.waterfall_source_frame_keys) == 17
        assert all(
            painted is expected
            for painted, expected in zip(
                view.waterfall_source_frame_keys,
                frames,
                strict=True,
            )
        )
    finally:
        view.close()


def test_prefix_overlay_curve_adds_only_the_new_item_when_range_is_stable(
    monkeypatch,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=4,
        selected_index=3,
        heavy_indices=(2, 3),
        plot_mode="Overlay",
    )
    frames = shell.navigation.frames
    shared_intensity = np.asarray(
        shell.scientific.traces[0].intensity,
        dtype=float,
    )
    traces = tuple(
        replace(trace, intensity=np.array(shared_intensity, copy=True))
        for trace in shell.scientific.traces
    )
    first_navigation = FrameNavigationProjection(
        frames,
        frames[2],
        frames[:3],
    )
    first_scientific = replace(
        shell.scientific,
        traces=traces[:3],
        heavy=replace(shell.scientific.heavy, frame=frames[2]),
    )
    view = ScientificView()
    try:
        _reconcile(
            view,
            first_scientific,
            first_navigation,
            completed=3,
            total=4,
        )
        old_items = tuple(view.curve.listDataItems())
        old_set_data_calls = 0

        for item in old_items:
            original = item.setData

            def counted(*args, _original=original, **kwargs):
                nonlocal old_set_data_calls
                old_set_data_calls += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(item, "setData", counted)

        _reconcile(
            view,
            replace(shell.scientific, traces=traces),
            shell.navigation,
            completed=4,
            total=4,
        )

        items = tuple(view.curve.listDataItems())
        assert old_set_data_calls == 0
        assert items[:3] == old_items
        assert len(items) == 4
        assert view._trace_history_keys == shell.navigation.selected
    finally:
        view.close()


@pytest.mark.parametrize("intensity_scale", ("Linear", "Sqrt", "Log"))
def test_average_sum_fold_reuses_exact_prefix_and_one_curve_item(
    monkeypatch,
    intensity_scale: str,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=5,
        selected_index=4,
        heavy_indices=(2, 4),
        plot_mode="Average",
    )
    frames = shell.navigation.frames
    traces = []
    for index, trace in enumerate(shell.scientific.traces):
        values = np.asarray(trace.intensity, dtype=float).copy()
        values += index - 2.0
        values[0] = np.nan
        values[3 + index] = np.nan
        values.setflags(write=False)
        traces.append(replace(trace, intensity=values))
    traces = tuple(traces)
    options = replace(
        shell.scientific.plot_options,
        intensity_scale=intensity_scale,
    )
    first_navigation, first_scientific = _prefix_state(
        shell,
        traces,
        3,
        plot_mode="Average",
        plot_options=options,
    )
    expected_average = _expected_aggregate(traces, "Average", intensity_scale)
    expected_sum = _expected_aggregate(traces, "Sum", intensity_scale)
    view = ScientificView()
    try:
        _reconcile(
            view,
            first_scientific,
            first_navigation,
            completed=3,
            total=5,
        )
        item = view.curve.listDataItems()[0]
        assert view._trace_aggregate_fold is not None
        assert len(view._trace_aggregate_fold.rows) == 3

        scale_calls = []
        set_data_calls = []
        original_scale = scientific_view_module._scaled_intensity
        original_set_data = item.setData

        def counted_scale(values, scale):
            scale_calls.append(values)
            return original_scale(values, scale)

        def counted_set_data(*args, **kwargs):
            set_data_calls.append(args)
            return original_set_data(*args, **kwargs)

        monkeypatch.setattr(
            scientific_view_module,
            "_scaled_intensity",
            counted_scale,
        )
        monkeypatch.setattr(item, "setData", counted_set_data)

        _reconcile(
            view,
            first_scientific,
            first_navigation,
            completed=3,
            total=5,
        )
        assert scale_calls == []
        assert set_data_calls == []
        assert view.curve.listDataItems()[0] is item

        suffix_scientific = replace(
            shell.scientific,
            traces=traces[3:],
            plot_mode="Average",
            plot_options=options,
        )
        _reconcile(
            view,
            suffix_scientific,
            shell.navigation,
            completed=5,
            total=5,
        )
        assert len(scale_calls) == 2
        assert scale_calls[0] is traces[3].intensity
        assert scale_calls[1] is traces[4].intensity
        assert len(set_data_calls) == 1
        assert view.curve.listDataItems()[0] is item
        fold = view._trace_aggregate_fold
        assert fold is not None and len(fold.rows) == 5
        np.testing.assert_array_equal(
            fold.projections["Average"].intensity,
            expected_average,
        )

        equal_values = traces[4].intensity.copy()
        equal_values.setflags(write=False)
        equal_current = replace(traces[4], intensity=equal_values)
        _reconcile(
            view,
            replace(suffix_scientific, traces=(equal_current,)),
            shell.navigation,
            completed=5,
            total=5,
        )
        assert len(scale_calls) == 2
        assert len(set_data_calls) == 1
        assert (
            view._trace_history_by_identity[id(frames[4])]
            is traces[4]
        )
        assert view._trace_aggregate_fold is fold

        _reconcile(
            view,
            replace(suffix_scientific, plot_mode="Sum"),
            shell.navigation,
            completed=5,
            total=5,
        )
        assert len(scale_calls) == 2
        assert scale_calls[0] is traces[3].intensity
        assert scale_calls[1] is traces[4].intensity
        assert len(set_data_calls) == 2
        assert view.curve.listDataItems()[0] is item
        assert item.name() == "Sum"
        assert view.legend.getLabel(item).text == "Sum"
        assert view._trace_aggregate_fold is fold
        np.testing.assert_array_equal(
            fold.projections["Sum"].intensity,
            expected_sum,
        )
        assert np.isnan(fold.projections["Average"].intensity[0])
        assert fold.projections["Sum"].intensity[0] == 0.0
    finally:
        view.close()


def test_average_fold_rebuilds_on_replacement_and_bypasses_writable_rows(
    monkeypatch,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=4,
        selected_index=3,
        heavy_indices=(0, 3),
        plot_mode="Average",
    )
    view = ScientificView()
    try:
        _reconcile(
            view,
            shell.scientific,
            shell.navigation,
            completed=4,
            total=4,
        )
        assert view._trace_aggregate_fold is not None

        changed_values = shell.scientific.traces[1].intensity.copy()
        changed_values[2] += 17.0
        changed_values.setflags(write=False)
        changed = replace(
            shell.scientific.traces[1],
            intensity=changed_values,
        )
        changed_traces = (
            shell.scientific.traces[0],
            changed,
            *shell.scientific.traces[2:],
        )
        sqrt_options = replace(
            shell.scientific.plot_options,
            intensity_scale="Sqrt",
        )
        writable_values = changed.intensity.copy()
        writable_values[4] += 1.0
        writable = replace(changed, intensity=writable_values)
        writable_traces = (
            changed_traces[0],
            writable,
            *changed_traces[2:],
        )
        positive_inf = writable_values.copy()
        negative_inf = changed_traces[2].intensity.copy()
        positive_inf[5] = np.inf
        negative_inf[5] = -np.inf
        positive_inf.setflags(write=False)
        negative_inf.setflags(write=False)
        infinite_traces = (
            replace(changed_traces[0], intensity=positive_inf),
            changed_traces[1],
            replace(changed_traces[2], intensity=negative_inf),
            changed_traces[3],
        )
        shifted_axis = changed_traces[1].axis.values.copy()
        shifted_axis += 0.01
        shifted_axis.setflags(write=False)
        mismatched = replace(
            changed_traces[1],
            axis=replace(changed_traces[1].axis, values=shifted_axis),
        )
        mismatched_traces = (
            changed_traces[0],
            mismatched,
            *changed_traces[2:],
        )
        cases = (
            (changed_traces, shell.scientific.plot_options, True),
            (changed_traces, sqrt_options, True),
            (writable_traces, shell.scientific.plot_options, False),
            (infinite_traces, shell.scientific.plot_options, True),
            (mismatched_traces, shell.scientific.plot_options, False),
        )
        scale_calls = []
        original_scale = scientific_view_module._scaled_intensity

        def counted_scale(values, scale):
            scale_calls.append(values)
            return original_scale(values, scale)

        monkeypatch.setattr(
            scientific_view_module,
            "_scaled_intensity",
            counted_scale,
        )
        for candidate, options, cache_expected in cases:
            scale_calls.clear()
            _reconcile(
                view,
                replace(
                    shell.scientific,
                    traces=candidate,
                    plot_options=options,
                ),
                shell.navigation,
                completed=4,
                total=4,
            )
            assert scale_calls == [trace.intensity for trace in candidate]
            fold = view._trace_aggregate_fold
            assert (fold is not None) is cache_expected
            scale = options.intensity_scale
            expected = _expected_aggregate(candidate, "Average", scale)
            actual = (
                fold.projections["Average"].intensity
                if fold is not None
                else view.curve.listDataItems()[0].yData
            )
            np.testing.assert_array_equal(actual, expected)
    finally:
        view.close()


def test_aggregate_matrix_preserves_one_shot_order_and_releases_on_mode_exit(
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=8,
        selected_index=7,
        heavy_indices=(1, 7),
        plot_mode="Average",
    )
    axis_values = _readonly([1.0])
    terms = (0.0, 0.0, -1.0, 1e259, -1.0, -1e259, 0.0, -1e99)
    traces = tuple(
        replace(
            trace,
            axis=replace(trace.axis, values=axis_values),
            intensity=_readonly([term]),
        )
        for trace, term in zip(shell.scientific.traces, terms, strict=True)
    )
    prefix_navigation, prefix_scientific = _prefix_state(shell, traces, 2)
    suffix_scientific = replace(shell.scientific, traces=traces[2:])
    view = ScientificView()
    try:
        _reconcile(
            view,
            prefix_scientific,
            prefix_navigation,
            completed=2,
            total=8,
        )
        assert view._trace_aggregate_fold.values.shape == (2, 1)

        _reconcile(
            view,
            suffix_scientific,
            shell.navigation,
            completed=8,
            total=8,
        )
        fold = view._trace_aggregate_fold
        assert fold is not None and fold.values.shape == (8, 1)
        expected_average = _expected_aggregate(traces, "Average")
        np.testing.assert_array_equal(
            fold.projections["Average"].intensity,
            expected_average,
        )

        _reconcile(
            view,
            replace(suffix_scientific, plot_mode="Sum"),
            shell.navigation,
            completed=8,
            total=8,
        )
        assert view._trace_aggregate_fold is fold
        expected_sum = _expected_aggregate(traces, "Sum")
        np.testing.assert_array_equal(
            fold.projections["Sum"].intensity,
            expected_sum,
        )

        full_values = weakref.ref(fold.values)
        _reconcile(
            view,
            prefix_scientific,
            prefix_navigation,
            completed=2,
            total=8,
        )
        assert view._trace_aggregate_fold is not fold
        del fold
        gc.collect()
        assert full_values() is None

        prefix_fold = view._trace_aggregate_fold
        prefix_values = weakref.ref(prefix_fold.values)
        _reconcile(
            view,
            replace(prefix_scientific, plot_mode="Overlay"),
            prefix_navigation,
            completed=2,
            total=8,
        )
        assert view._trace_aggregate_fold is None
        del prefix_fold
        gc.collect()
        assert prefix_values() is None
    finally:
        view.close()


def test_aggregate_matrix_promotes_dtype_and_publishes_only_after_reduction(
    monkeypatch,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(
        frame_count=3,
        selected_index=2,
        heavy_indices=(1, 2),
        plot_mode="Average",
    )
    traces = tuple(
        replace(
            trace,
            intensity=_readonly(
                trace.intensity,
                dtype=np.float32 if index < 2 else np.float64,
            ),
        )
        for index, trace in enumerate(shell.scientific.traces)
    )
    prefix_navigation, prefix_scientific = _prefix_state(shell, traces, 2)
    suffix_scientific = replace(shell.scientific, traces=(traces[2],))
    view = ScientificView()
    try:
        _reconcile(
            view,
            prefix_scientific,
            prefix_navigation,
            completed=2,
            total=3,
        )
        prior = view._trace_aggregate_fold
        assert prior is not None and prior.values.dtype == np.dtype(np.float32)

        original_nanmean = scientific_view_module.np.nanmean

        def fail_reduction(*_args, **_kwargs):
            raise RuntimeError("injected aggregate reduction failure")

        monkeypatch.setattr(
            scientific_view_module.np,
            "nanmean",
            fail_reduction,
        )
        with pytest.raises(
            RuntimeError,
            match="injected aggregate reduction failure",
        ):
            _reconcile(
                view,
                suffix_scientific,
                shell.navigation,
                completed=3,
                total=3,
            )
        assert view._trace_aggregate_fold is prior

        monkeypatch.setattr(
            scientific_view_module.np,
            "nanmean",
            original_nanmean,
        )
        _reconcile(
            view,
            suffix_scientific,
            shell.navigation,
            completed=3,
            total=3,
        )
        fold = view._trace_aggregate_fold
        assert fold is not None and fold.values.dtype == np.dtype(np.float64)
        expected = _expected_aggregate(traces, "Average")
        np.testing.assert_array_equal(
            fold.projections["Average"].intensity,
            expected,
        )
    finally:
        view.close()


def test_single_skips_unused_offset_scan_but_stacked_curve_retains_it(
    monkeypatch,
) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original = scientific_view_module._overlay_step
    single = make_shell_projection(
        frame_count=1,
        selected_index=0,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    single_view = ScientificView()
    try:
        monkeypatch.setattr(
            scientific_view_module,
            "_overlay_step",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "Single presentation computed an unused overlay offset"
                )
            ),
        )
        _reconcile(
            single_view,
            single.scientific,
            single.navigation,
            completed=1,
            total=1,
        )
        _reconcile(
            single_view,
            single.scientific,
            single.navigation,
            completed=1,
            total=1,
        )
    finally:
        single_view.close()

    stacked = make_shell_projection(
        frame_count=2,
        selected_index=1,
        heavy_indices=(0, 1),
        plot_mode="Overlay",
    )
    stacked_view = ScientificView()
    calls = []

    def counted(traces, offset_percent):
        calls.append((traces, offset_percent))
        return original(traces, offset_percent)

    monkeypatch.setattr(scientific_view_module, "_overlay_step", counted)
    try:
        _reconcile(
            stacked_view,
            stacked.scientific,
            stacked.navigation,
            completed=2,
            total=2,
        )
        assert len(calls) == 1
        assert len(calls[0][0]) == len(stacked.scientific.traces)
        assert all(
            actual.frame is expected.frame
            for actual, expected in zip(
                calls[0][0],
                stacked.scientific.traces,
                strict=True,
            )
        )
        assert calls[0][1] == stacked.scientific.plot_options.overlay_offset
    finally:
        stacked_view.close()


def test_projection_marks_only_active_lifecycle_as_live_update() -> None:
    navigation = FrameNavigationProjection()
    preferences = ScientificPreferences(plot_mode="Overlay")

    running = build_scientific_projection(
        (),
        navigation,
        frozenset(),
        preferences,
        "",
        RunPhase.RUNNING,
    )
    terminal = build_scientific_projection(
        (),
        navigation,
        frozenset(),
        preferences,
        "",
        RunPhase.FINALIZING,
    )
    failed = build_scientific_projection(
        (),
        navigation,
        frozenset(),
        preferences,
        "",
        RunPhase.FAILED,
    )

    assert running.live_update is True
    assert terminal.live_update is False
    assert failed.live_update is False


def test_3621_runtime_projection_is_two_phase_and_prefix_delta_only(
    monkeypatch,
) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    runtime = controller._runtime
    display = acquisition.publication_store
    for label in range(2, 3622):
        delta = display.append_navigation(
            "run.a",
            "/out/a.nxs",
            label,
        )
        assert runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = runtime.navigation.frames
    assert len(frames) == 3621
    assert runtime.select_navigation(frames[-1], frames)

    requested = []

    def record_request(
        _projection,
        request,
        *_owners,
    ):
        requested.append(request.frame)
        return None

    monkeypatch.setattr(runtime, "resolve_projection", record_request)
    preferences = ScientificPreferences(plot_mode="Overlay")
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert tuple(requested[:-1]) == frames
    assert requested[-1] is frames[-1]
    assert controller.commit_navigation_projection(frames)

    appended = display.append_navigation(
        "run.a",
        "/out/a.nxs",
        3622,
    )
    assert runtime.accept_navigation(appended, plot_mode="Overlay")
    target = runtime.navigation.selected
    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [appended.appended, appended.appended]

    # A failed shell application performs no acknowledgement.  Retrying must
    # resolve the same exact delta instead of silently losing that row.
    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [appended.appended, appended.appended]
    assert controller.commit_navigation_projection(target)

    # Render-only changes reuse retained 1-D history and read only the current
    # heavy anchor.  A trace-projection axis change is a full reseed.
    requested.clear()
    controller.project_navigation(
        preferences=replace(preferences, color_map="magma"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [appended.appended, appended.appended]

    requested.clear()
    controller.project_navigation(
        preferences=replace(preferences, plot_mode="Waterfall"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [appended.appended, appended.appended]

    requested.clear()
    controller.project_navigation(
        preferences=replace(preferences, plot_mode="Average"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [target[-1], target[-1]]
    assert controller.commit_navigation_projection(target)

    # Average, Sum, Overlay, and Waterfall share one numeric trace scope.
    # Switching presentation modes therefore retains the acknowledged
    # history and resolves only the current heavy anchor.
    requested.clear()
    controller.project_navigation(
        preferences=replace(preferences, plot_mode="Sum"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [target[-1], target[-1]]
    assert controller.commit_navigation_projection(target)

    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [target[-1], target[-1]]
    assert controller.commit_navigation_projection(target)

    requested.clear()
    controller.project_navigation(
        preferences=replace(preferences, plot_axis="2theta"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert tuple(requested[:-1]) == target
    assert requested[-1] is target[-1]

    equal_distinct = replace(appended.appended)
    assert equal_distinct == appended.appended
    assert equal_distinct is not appended.appended
    assert runtime.owns_frame(equal_distinct) is False


def test_3621_sparse_membership_delta_projects_only_changed_identities(
    monkeypatch,
) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    runtime = controller._runtime
    display = acquisition.publication_store
    for label in range(2, 3622):
        delta = display.append_navigation(
            "run.a",
            "/out/a.nxs",
            label,
        )
        assert runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = runtime.navigation.frames
    assert runtime.select_navigation(frames[-1], frames)

    requested = []

    def record_request(
        _projection,
        request,
        *_owners,
    ):
        requested.append(request.frame)
        return None

    monkeypatch.setattr(runtime, "resolve_projection", record_request)
    preferences = ScientificPreferences(plot_mode="Overlay")
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert controller.commit_navigation_projection(frames)

    removed = (frames[100], frames[1800], frames[-2])
    removed_ids = {id(frame) for frame in removed}
    target = tuple(
        frame for frame in frames if id(frame) not in removed_ids
    )
    assert runtime.select_navigation(frames[-1], target)
    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [frames[-1], frames[-1]]
    assert controller.commit_navigation_projection(target)

    readded = removed[1]
    readded_target = tuple(
        frame
        for frame in frames
        if frame is readded or id(frame) not in removed_ids
    )
    assert runtime.select_navigation(frames[-1], readded_target)
    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert requested == [readded, frames[-1], frames[-1]]


def test_3621_excluded_current_prunes_history_without_reseed(monkeypatch) -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    runtime = controller._runtime
    for label in range(2, 3622):
        delta = acquisition.publication_store.append_navigation(
            "run.a", "/out/a.nxs", label
        )
        assert runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = runtime.navigation.frames
    generation = controller.selection.display_generation
    payload_by_id = {
        id(frame): StandardDisplayPayload(
            generation, frame, str(frame.local_frame_label),
            _view(frame.local_frame_label, float(frame.local_frame_label)),
        )
        for frame in frames
    }
    requested = []
    def resolve(_projection, request, *_owners):
        requested.append(request.frame); return payload_by_id[id(request.frame)]
    monkeypatch.setattr(runtime, "resolve_projection", resolve)
    preferences = ScientificPreferences(plot_mode="Overlay")
    view = ScientificView()
    try:
        def project_render_commit():
            payloads = controller.project_navigation(preferences=preferences)
            scientific = build_scientific_projection(
                payloads, runtime.navigation, frozenset(frames), preferences, "", RunPhase.RUNNING,
            )
            _reconcile(view, scientific, runtime.navigation, completed=3621, total=3621)
            assert controller.commit_navigation_projection(view._trace_history_keys)
            return scientific

        project_render_commit()
        assert len(view._trace_history_keys) == 3621
        prior_by_id = dict(view._trace_history_by_identity)
        target = frames[:-1]
        assert runtime.select_navigation(frames[-1], target)
        requested.clear()
        removal = project_render_commit()
        assert requested == [frames[-1]]
        assert removal.traces == ()
        assert removal.heavy is not None and removal.heavy.frame is frames[-1]
        assert removal.retain_display is False
        assert view._trace_history_keys == target
        assert all(
            trace is prior_by_id[id(frame)]
            for frame, trace in zip(target, view._trace_history_by_identity.values(), strict=True)
        )
        retained = tuple(view._trace_history_by_identity.values())
        requested.clear()
        unchanged = project_render_commit()
        assert unchanged.traces == () and requested == [frames[-1]]
        assert tuple(view._trace_history_by_identity.values()) == retained
    finally:
        view.close()
