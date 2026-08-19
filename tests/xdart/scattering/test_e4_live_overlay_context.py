"""Focused live-selection contract for multi-trace scientific modes."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from tests.xdart.scattering.test_e2lv_live_display import (
    _mounted,
    _shell_diagnostic,
    _standard_page,
    _wait,
)
from tests.xdart.scattering.test_e3_context_contract import (
    _running_controller,
)
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    DisplayNavigationDelta,
)
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase


_HISTORY_MODES = ("Overlay", "Waterfall", "Average", "Sum")


def _assert_exact(
    actual: tuple[DisplayFrameKey, ...],
    expected: tuple[DisplayFrameKey, ...],
) -> None:
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected))


class _HostileIdentityIndex(dict[int, DisplayFrameKey]):
    """Exact index whose existing contents may not be scanned or copied."""

    def __iter__(self):
        raise AssertionError("live append rebuilt the acquisition identity index")

    def keys(self):
        raise AssertionError("live append scanned acquisition identity keys")

    def items(self):
        raise AssertionError("live append scanned acquisition identity items")

    def values(self):
        raise AssertionError("live append scanned acquisition identity values")


def test_live_delta_updates_identity_index_in_place_without_full_rebuild() -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    runtime = controller._runtime
    display = acquisition.publication_store
    assert display.catalog.resize(2) == ()
    first = controller.navigation.current
    assert first is not None
    second_delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    assert controller.accept_navigation(second_delta, plot_mode="Overlay")
    second = second_delta.appended

    hostile = _HostileIdentityIndex()
    dict.__setitem__(hostile, id(first), first)
    dict.__setitem__(hostile, id(second), second)
    runtime._acquisition_frame_by_id = hostile
    third_delta = display.append_navigation("run.a", "/out/a.nxs", 3)
    assert third_delta.retired == (first,)

    assert controller.accept_navigation(third_delta, plot_mode="Overlay")

    assert runtime._acquisition_frame_by_id is hostile
    assert len(hostile) == 2
    assert hostile.get(id(first)) is None
    assert hostile.get(id(second)) is second
    assert hostile.get(id(third_delta.appended)) is third_delta.appended
    _assert_exact(controller.navigation.frames, (second, third_delta.appended))
    assert controller.owns_frame(first) is False
    assert controller.owns_frame(second) is True
    assert controller.owns_frame(third_delta.appended) is True


def test_live_delta_invalidates_full_demand_only_when_current_changes(
    monkeypatch,
) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    display = acquisition.publication_store
    first = controller.navigation.current
    assert first is not None
    invalidations: list[dict[str, object]] = []
    monkeypatch.setattr(
        display,
        "invalidate_full_demand",
        lambda **kwargs: invalidations.append(kwargs),
    )

    held_delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    assert controller.accept_navigation(
        held_delta,
        plot_mode="Overlay",
        follow_latest=False,
    )
    assert controller.navigation.current is first
    assert invalidations == []

    followed_delta = display.append_navigation("run.a", "/out/a.nxs", 3)
    assert controller.accept_navigation(followed_delta, plot_mode="Overlay")
    assert controller.navigation.current is followed_delta.appended
    assert invalidations == [{}]


def test_foreign_and_stale_live_deltas_leave_navigation_index_unchanged() -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    runtime = controller._runtime
    display = acquisition.publication_store
    identity = controller.run_identity
    assert identity is not None
    original_navigation = controller.navigation
    original_index = runtime._acquisition_frame_by_id
    original_frame = original_navigation.current
    assert original_frame is not None

    foreign_identity = RunIdentity(
        identity.generation + 1,
        f"{identity.fingerprint}-foreign",
    )
    foreign = DisplayFrameKey(
        foreign_identity,
        "run.a",
        "/out/a.nxs",
        90,
        90,
    )
    assert controller.accept_navigation(
        DisplayNavigationDelta(foreign),
        plot_mode="Overlay",
    ) is False

    assert display.catalog.resize(1) == ()
    stale_delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    latest_delta = display.append_navigation("run.a", "/out/a.nxs", 3)
    assert stale_delta.appended in latest_delta.retired
    assert controller.accept_navigation(stale_delta, plot_mode="Overlay") is False

    assert controller.navigation is original_navigation
    assert runtime._acquisition_frame_by_id is original_index
    assert len(original_index) == 1
    assert original_index.get(id(original_frame)) is original_frame


@pytest.mark.parametrize("plot_mode", _HISTORY_MODES)
def test_live_multitrace_delta_preserves_exact_selected_history_through_cap(
    plot_mode: str,
) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    display = acquisition.publication_store
    assert display.catalog.resize(4) == ()
    first = controller.navigation.current
    assert first is not None

    second_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 2
    )
    assert controller.accept_navigation(
        second_delta,
        plot_mode=plot_mode,
    )
    second = second_delta.appended
    third_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 3
    )
    assert controller.accept_navigation(
        third_delta,
        plot_mode=plot_mode,
    )
    third = third_delta.appended
    _assert_exact(
        controller.navigation.selected,
        (first, second, third),
    )

    assert controller.select_navigation(third, (first, third))
    fourth_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 4
    )
    assert fourth_delta.retired == ()
    assert controller.accept_navigation(
        fourth_delta,
        plot_mode=plot_mode,
    )
    fourth = fourth_delta.appended
    _assert_exact(
        controller.navigation.selected,
        (first, third, fourth),
    )

    fifth_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 5
    )
    assert fifth_delta.retired == (first,)
    assert controller.accept_navigation(
        fifth_delta,
        plot_mode=plot_mode,
    )
    fifth = fifth_delta.appended
    _assert_exact(
        controller.navigation.selected,
        (third, fourth, fifth),
    )

    sixth_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 6
    )
    assert sixth_delta.retired == (second,)
    assert controller.accept_navigation(
        sixth_delta,
        plot_mode=plot_mode,
    )
    sixth = sixth_delta.appended
    _assert_exact(
        controller.navigation.frames,
        (third, fourth, fifth, sixth),
    )
    _assert_exact(
        controller.navigation.selected,
        (third, fourth, fifth, sixth),
    )
    assert controller.navigation.current is sixth


def test_live_single_delta_keeps_only_the_exact_newest_key() -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    display = acquisition.publication_store
    first = controller.navigation.current
    assert first is not None

    delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    assert controller.accept_navigation(delta, plot_mode="Single")
    _assert_exact(controller.navigation.frames, (first, delta.appended))
    assert controller.navigation.current is delta.appended
    _assert_exact(controller.navigation.selected, (delta.appended,))


@pytest.mark.parametrize("plot_mode", _HISTORY_MODES)
def test_reenabling_auto_last_moves_only_current_and_keeps_exclusions(
    plot_mode: str,
) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = (
        _running_controller()
    )
    display = acquisition.publication_store
    first = controller.navigation.current
    assert first is not None
    second_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 2
    )
    third_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 3
    )
    assert controller.accept_navigation(
        second_delta, plot_mode=plot_mode
    )
    assert controller.accept_navigation(
        third_delta, plot_mode=plot_mode
    )
    assert controller.select_navigation(first, (first,))

    assert controller.select_latest_navigation(plot_mode=plot_mode)

    assert controller.navigation.current is third_delta.appended
    _assert_exact(
        controller.navigation.selected,
        (first,) if plot_mode in {"Overlay", "Waterfall"} else controller.navigation.frames,
    )
    fourth_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 4
    )
    assert controller.accept_navigation(
        fourth_delta, plot_mode=plot_mode
    )
    _assert_exact(
        controller.navigation.selected,
        (first, fourth_delta.appended) if plot_mode in {"Overlay", "Waterfall"}
        else controller.navigation.frames,
    )


def test_show_all_clears_exclusions_and_auto_last_preserves_them() -> None:
    controller, _lifecycle, _executor, _loader, acquisition = _running_controller()
    display = acquisition.publication_store
    first = controller.navigation.current
    assert first is not None
    second_delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    third_delta = display.append_navigation("run.a", "/out/a.nxs", 3)
    assert controller.accept_navigation(second_delta, plot_mode="Overlay")
    assert controller.accept_navigation(third_delta, plot_mode="Overlay")
    second = second_delta.appended
    third = third_delta.appended
    assert controller.select_navigation(first, (first, third))

    assert controller.select_latest_navigation(plot_mode="Overlay")
    assert controller.navigation.current is third
    _assert_exact(controller.navigation.selected, (first, third))

    events = []
    owner = SimpleNamespace(
        _closing=False,
        _closed=False,
        _shell=SimpleNamespace(
            browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: events.append(
                    "cancel"
                )
            )
        ),
        _context_controller=controller,
        _refresh_shell=lambda: events.append("refresh"),
    )
    ScatteringWorkspace._handle_shell_command(
        owner,
        ShellCommand(ShellCommandKind.SHOW_ALL),
    )
    _assert_exact(
        controller.navigation.selected,
        (first, second, third),
    )
    assert events == ["cancel", "refresh"]


@pytest.mark.parametrize("plot_mode", _HISTORY_MODES)
def test_auto_last_off_keeps_mode_specific_arrival_contract(plot_mode: str) -> None:
    controller, _lifecycle, _executor, _loader, acquisition = _running_controller()
    display = acquisition.publication_store
    first = controller.navigation.current
    assert first is not None
    second_delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    third_delta = display.append_navigation("run.a", "/out/a.nxs", 3)
    assert controller.accept_navigation(
        second_delta, plot_mode=plot_mode
    )
    assert controller.accept_navigation(
        third_delta, plot_mode=plot_mode
    )
    assert controller.select_navigation(
        first,
        (first, third_delta.appended),
    )

    fourth_delta = display.append_navigation(
        "run.a", "/out/a.nxs", 4
    )
    assert controller.accept_navigation(
        fourth_delta,
        plot_mode=plot_mode,
        follow_latest=False,
    )

    assert controller.navigation.current is first
    _assert_exact(
        controller.navigation.selected,
        (first, third_delta.appended, fourth_delta.appended) if plot_mode in {"Overlay", "Waterfall"}
        else (first, third_delta.appended),
    )


@pytest.mark.parametrize("plot_mode", _HISTORY_MODES)
def test_mounted_live_multitrace_mode_renders_every_sequential_delta(
    monkeypatch,
    tmp_path,
    plot_mode: str,
) -> None:
    qapp, page, lifecycle, _executor, _output = _standard_page(
        monkeypatch,
        tmp_path,
        labels=(1, 2, 3),
    )
    shell, controller = _mounted(page)
    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        shell.scientific.plot_mode.setCurrentText(plot_mode)
        _wait(qapp, lambda: page._preferences.plot_mode == plot_mode)
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: lifecycle.phase is RunPhase.IDLE,
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )

        frames = controller.navigation.frames
        assert len(frames) == 3
        _assert_exact(controller.navigation.selected, frames)
        expected_curve_count = (
            3 if plot_mode in {"Overlay", "Waterfall"} else 1
        )
        _wait(
            qapp,
            lambda: (
                len(shell.scientific.curve.listDataItems())
                == expected_curve_count
            ),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
    finally:
        page.close_workspace()
        page.close()
