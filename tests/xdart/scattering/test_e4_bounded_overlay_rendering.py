from __future__ import annotations

from dataclasses import replace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering import scientific_view as scientific_view_module
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
from xdart.gui.tabs.scattering.state_machine import RunPhase

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_e3_context_contract import _running_controller


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

        _reconcile(
            view,
            delta_scientific,
            shell.navigation,
            completed=17,
            total=20,
        )

        assert calls == {"scale": 0, "stack": 0}
        assert view._trace_history_keys == shell.navigation.selected
        assert view.waterfall.image.image.shape == (64, 16)

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
        _browse_hydration_owner=None,
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
        preferences=replace(preferences, plot_mode="Average"),
        processing_mode="Int 2D",
        live_update=True,
    )
    assert tuple(requested[:-1]) == target
    assert requested[-1] is target[-1]

    # Re-entering an accumulating mode cannot reuse the ledger of the view
    # that Average just replaced.
    requested.clear()
    controller.project_navigation(
        preferences=preferences,
        processing_mode="Int 2D",
        live_update=True,
    )
    assert tuple(requested[:-1]) == target
    assert requested[-1] is target[-1]

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
