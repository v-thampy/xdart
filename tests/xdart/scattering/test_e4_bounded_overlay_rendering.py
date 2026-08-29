from __future__ import annotations

from dataclasses import replace

import numpy as np
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
