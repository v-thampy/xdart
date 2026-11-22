"""Focused GUI contract for the P3 display-background operation."""

from __future__ import annotations

import ast
import weakref
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.context_projection import apply_presentation_background
from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp, OperationTerminalStatus,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.presentation_background import (
    DisplayBackgroundRendererReleaseReceipt, PresentationBackgroundOwner,
    prepare_background_plan,
)
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, FrameNavigationProjection, ShellCommand, ShellCommandKind, SlicePin, TraceProjection,
)
from xrd_tools.reduction import DisplayBackgroundPlan
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_e3_context_contract import (
    _acquisition, _running_controller, _view,
)


def _inputs():
    axis = np.arange(3.0)
    first = np.array([1.0, np.nan, 5.0])
    second = np.array([3.0, 7.0, np.nan])
    plan = DisplayBackgroundPlan(
        "integrated_1d", ("first", "second"), ((3,), (3,)),
        (((3,),), ((3,),)), (("q\0A^-1",), ("q\0A^-1",)))
    return plan, ((first, axis), (second, axis.copy()))


def _root(value):
    while isinstance(value, np.ndarray): value = value.base
    return value


def _has_array(value) -> bool:
    if isinstance(value, np.ndarray): return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_has_array(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, (tuple, list, dict)):
        items = value.items() if isinstance(value, dict) else value
        return any(_has_array(item) for item in items)
    return False


def _reserve(owner, plan, contributors, *, keys=(11, 12), indices=(0, 1)):
    return owner.reserve(
        plan, contributors, stamp=OperationContextStamp(0, "context", 1),
        active_key=("context", 1, plan.domain, "Int 1D", "integrated"),
        projection_keys=keys, projection_indices=indices)


def _assert_background_released(owner) -> None:
    assert owner.phase == "RELEASED"
    assert owner.projection() is None
    assert owner._contributors == () and owner._result is None
    assert owner._display_values == ()
    assert owner.reserved_bytes == owner.active_bytes == 0
    assert all(getattr(owner, name) is None for name in (
        "_plan", "_stamp", "_active_key", "_reservation"))


def _renderer_roles(view) -> tuple[tuple[str, object], ...]:
    roles = []
    for name in ("raw", "cake", "waterfall"):
        pane = getattr(view, name)
        for member in ("raw_image", "displayed_image"):
            value = getattr(pane.canvas, member, None)
            roles.append((f"{name}.{member}",
                          None if not isinstance(value, np.ndarray) else value.shape))
        image = getattr(pane.image, "image", None)
        roles.append((f"{name}.image",
                      None if not isinstance(image, np.ndarray) else image.shape))
    roles.append(("curve.xy", tuple(
        (item.xData.shape, item.yData.shape)
        for item in view.curve.listDataItems())))
    return tuple(roles)


def _active_page_background():
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    page._context_controller._runtime.adopt_acquisition(identity, acquisition)
    page._refresh_shell()
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
    worker = page._workspace_operations._slot._worker
    assert worker is not None; worker.join(3); assert not worker.is_alive()
    page._drain_executor(); qapp.processEvents()
    assert page._background_owner.phase == "ACTIVE"
    return qapp, page, acquisition


def _seed_page_1d_background(page: ScatteringWorkspace) -> PresentationBackgroundOwner:
    shell = make_shell_projection(frame_count=2, selected_index=1,
                                  plot_mode="Overlay")
    state = replace(shell.scientific, processing_mode="Int 1D")
    contributors = tuple((trace.intensity, trace.axis.values)
                         for trace in state.traces)
    plan = DisplayBackgroundPlan(
        "integrated_1d", tuple(f"trace-{index}" for index in range(2)),
        tuple(item[0].shape for item in contributors),
        tuple((item[1].shape,) for item in contributors),
        tuple(("Q\0Å⁻¹",) for _item in contributors))
    owner = page._background_owner
    key = ("context", 1, "integrated_1d", "Int 1D")
    reservation = owner.reserve(
        plan, contributors, stamp=OperationContextStamp(0, "context", 1),
        active_key=key, projection_keys=tuple(id(trace.frame) for trace in state.traces),
        projection_indices=(0, 1))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, tuple((*item, 1.0) for item in contributors))
    page._shell.scientific.expect_display_background(key)
    active = apply_presentation_background(state, owner.projection())
    page._shell.scientific.reconcile(
        active, shell.navigation, completed=2, total=2, detail="Ready")
    return owner


def test_parent_red_empty_selection_refuses_set_background() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    before = page._preferences
    try:
        page._handle_shell_command(
            ShellCommand(ShellCommandKind.SET_BACKGROUND)
        )
        assert (
            page._preferences is before
            and "Select at least one display frame" in page._notice_text
        )
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_owner_exact_accounting_bytes_roots_zero_copy_stage_and_d_once(monkeypatch) -> None:
    from xdart.gui.tabs.scattering import presentation_background as owner_module

    plan, contributors = _inputs()
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    assert owner.capacity_bytes == 536_870_912
    # C=96, V=3, X=24, D=48, T=589824.
    q = 96 + 32 * 3 + 24 + 48 + 589_824
    owner._capacity = q
    reservation = _reserve(owner, plan, contributors)
    assert type(reservation) is int and owner.phase == "RESERVED"
    assert owner.reserved_bytes == q
    assert all(isinstance(_root(array), bytes) and not array.flags.writeable
               for row in owner._contributors for array in row)
    assert all(not np.shares_memory(source, copied)
               for row, copied_row in zip(contributors, owner._contributors, strict=True)
               for source, copied in zip(row, copied_row, strict=True))
    c_roots = tuple(id(_root(array)) for row in owner._contributors for array in row)
    assert len(set(c_roots)) == 4 and owner._result is None and owner._display_values == ()
    captured, runner = [], owner_module.run_display_background
    def observe(*args, **kwargs):
        captured.append(runner(*args, **kwargs)); return captured[-1]
    monkeypatch.setattr(owner_module, "run_display_background", observe)
    contributors[0][0][0] = 99.0
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert len(captured) == 1 and owner._result is captured[0]
    assert owner.phase == "STAGED" and not _has_array(receipt)
    assert receipt.retained_bytes == 72 == owner.active_bytes
    r_roots = (id(_root(owner._result.values)), id(_root(owner._result.finite_counts)),
               *(id(_root(axis)) for axis in owner._result.axes))
    assert owner._contributors == () and len(set(r_roots)) == 3
    assert owner.promote(receipt, tuple((*item, 1.0) for item in contributors))
    projection = owner.projection(); assert projection is not None
    displayed = tuple(value for _key, value in projection[1])
    np.testing.assert_allclose(displayed[0], [97.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(displayed[1], [1.0, 0.0, np.nan], equal_nan=True)
    assert all(value.base is None and not value.flags.writeable for value in displayed)
    d_roots = tuple(id(value) for value in displayed)
    assert len(set(d_roots)) == 2 and not set(d_roots) & set(r_roots)
    assert owner.phase == "ACTIVE" and owner.active_bytes == 72 + 48
    assert owner.projection()[1][0][1] is displayed[0]
    assert tuple(id(value) for _key, value in owner.projection()[1]) == d_roots
    assert (id(_root(owner._result.values)), id(_root(owner._result.finite_counts)),
            *(id(_root(axis)) for axis in owner._result.axes)) == r_roots
    assert not owner.release(DisplayBackgroundRendererReleaseReceipt(
        receipt.active_key, False))
    assert owner.release(DisplayBackgroundRendererReleaseReceipt(
        receipt.active_key, True)) and owner.phase == "RELEASED"
    _assert_background_released(owner)


def test_noncontiguous_image_target_uses_only_the_charged_d_root() -> None:
    background = np.arange(12.0).reshape(3, 4)
    y_axis, x_axis = np.arange(3.0), np.arange(4.0)
    plan = DisplayBackgroundPlan(
        "integrated_2d", ("image",), ((3, 4),), (((3,), (4,)),),
        (("q\0A^-1", "chi\0degree"),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = _reserve(
        owner, plan, ((background, y_axis, x_axis),), keys=(11,), indices=(0,))
    receipt = owner.run_and_stage(reservation, lambda: False)
    source = (np.arange(12.0).reshape(4, 3).T + 20.0)
    source_before = source.copy()
    assert not source.flags.c_contiguous
    assert owner.promote(receipt, ((source, y_axis, x_axis),))
    displayed = owner.projection()[1][0][1]
    np.testing.assert_allclose(displayed, source - owner._result.values)
    np.testing.assert_array_equal(source, source_before)
    assert displayed.base is None and not np.shares_memory(displayed, source)
    assert owner.active_bytes == receipt.retained_bytes + displayed.nbytes


def test_q_equality_passes_and_one_byte_over_refuses_before_copy(monkeypatch) -> None:
    from xdart.gui.tabs.scattering import presentation_background as owner_module

    plan, contributors = _inputs()
    q = 96 + 32 * 3 + 24 + 48 + 589_824
    admitted = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    admitted._capacity = q
    assert _reserve(admitted, plan, contributors) == 1
    admitted.abort(1, "TEST_RELEASE")
    copies, copier = [], owner_module._immutable_copy
    def observed(value):
        copies.append(value); return copier(value)
    monkeypatch.setattr(owner_module, "_immutable_copy", observed)
    refused = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    refused._capacity = q - 1
    assert _reserve(refused, plan, contributors) is None
    assert copies == [] and refused.phase == "EMPTY"


def test_reserved_and_staged_clear_wait_for_worker_finalizer() -> None:
    plan, contributors = _inputs()
    reserved = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    token = _reserve(reserved, plan, contributors)
    roots = reserved._contributors
    assert token and not reserved.release() and reserved.phase == "CLEANUP_PENDING"
    assert reserved._contributors is roots
    reserved.finalize(token, "CANCELLED")
    assert reserved.phase == "RELEASED" and reserved._contributors == ()

    staged = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    token = _reserve(staged, plan, contributors)
    receipt = staged.run_and_stage(token, lambda: False)
    result = staged._result
    assert not staged.release() and staged.phase == "CLEANUP_PENDING"
    assert staged._result is result
    staged.finalize(token, "TRANSFERRED")
    assert staged.phase == "RELEASED" and staged._result is None


def test_one_global_reservation_and_mixed_axis_units_refuse() -> None:
    plan, contributors = _inputs()
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    token = _reserve(owner, plan, contributors)
    assert token and _reserve(owner, plan, contributors) is None
    owner.abort(token, "REPLACED")
    assert _reserve(owner, plan, contributors) == token + 1
    owner.abort(token + 1, "DONE")

    payloads = []
    current = object()
    for index, unit in enumerate(("A^-1", "nm^-1")):
        frame = current if index == 0 else object()
        key = type("Key", (), {"source_scan": "scan", "artifact": "a",
                                "local_frame_label": index})()
        if index == 0: current = key
        axis = type("Axis", (), {"values": np.arange(3.0), "label": "q",
                                 "unit": unit})()
        view = type("View", (), {"axis_1d": axis,
                    "intensity_1d": np.ones(3) * (index + 1)})()
        payloads.append(type("Payload", (), {"view": view, "frame_key": key})())
    assert prepare_background_plan(tuple(payloads), "integrated_1d", current) is None


@pytest.mark.parametrize("outcome", ("stale", "failed"))
def test_background_stale_and_failure_finalize_exact_roots(
    monkeypatch, outcome: str,
) -> None:
    from xdart.gui.tabs.scattering import presentation_background as owner_module

    plan, contributors = _inputs()
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = _reserve(owner, plan, contributors)
    entered, release = Event(), Event()
    runner = owner_module.run_display_background
    def controlled(*args, **kwargs):
        entered.set(); assert release.wait(2)
        if outcome == "failed": raise RuntimeError("background failed")
        return runner(*args, **kwargs)
    monkeypatch.setattr(owner_module, "run_display_background", controlled)
    slot = OperationSlot()
    identity = slot.begin_background(
        plan, OperationContextStamp(0, "context", 1), owner, reservation)
    assert identity is not None and entered.wait(2)
    if outcome == "stale":
        slot.observe_stamp(OperationContextStamp(0, "other", 2))
    release.set(); worker = slot._worker; worker.join(2)
    update = slot.poll(identity)
    assert update is not None and update.terminal is not None
    if outcome == "stale":
        assert update.stale and update.terminal.status is OperationTerminalStatus.RETURNED
    else:
        assert not update.stale and update.terminal.status is OperationTerminalStatus.FAILED
    _assert_background_released(owner)


@pytest.mark.parametrize("drift", ("generation", "membership", "shape"))
def test_pre_adoption_context_membership_and_shape_drift_release_exact_roots(
    monkeypatch, drift: str,
) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    runtime = page._context_controller._runtime
    runtime.adopt_acquisition(identity, acquisition); page._refresh_shell()
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
    worker = page._workspace_operations._slot._worker
    assert worker is not None; worker.join(3); assert not worker.is_alive()
    if drift == "generation":
        runtime._selection = replace(runtime._selection,
                                     display_generation=runtime._selection.display_generation + 1)
    elif drift == "membership":
        runtime._acquisition_navigation = replace(
            runtime._acquisition_navigation, selected=())
    else:
        payload = page._context_controller.project_background_contributors()[0]
        changed = replace(
            payload.view, intensity_2d=np.zeros((3, 3)),
            axis_2d_y=replace(payload.view.axis_2d_y, values=np.arange(3.0)))
        monkeypatch.setattr(
            page._context_controller, "project_background_contributors",
            lambda *_args: (replace(payload, view=changed),))
    try:
        page._drain_executor(); qapp.processEvents()
        _assert_background_released(page._background_owner)
        assert "not adopted" in page._notice_text or "context changed" in page._notice_text
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize("entry", ("context", "mode", "close"))
def test_active_background_revokes_at_every_page_boundary(monkeypatch, entry: str) -> None:
    """An Int 2D background goes at a mode change and at close. Since 1.2.18 a scan or file choice keeps it, as a
    frame choice does (his picks "Int 2D follows frames too" and "Any scan or file"; his words: "If it is dropped on
    selection change what is the point of it at all?"), and so does a detector-mode change, which changes only the
    raw image (test_a_detector_mode_change_keeps_an_int_background)."""
    qapp, page, _acquisition_context = _active_page_background()
    delegated = []
    try:
        current = page._context_controller.navigation.current
        if entry == "selection":
            monkeypatch.setattr(page, "_select_frames",
                                lambda _command: delegated.append(page._background_owner.phase))
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SELECT_FRAME, frame=current, frames=(current,)))
        elif entry == "context":
            def select_scan(_value, *, is_directory):
                assert is_directory is False
                delegated.append(page._background_owner.phase)
            monkeypatch.setattr(page, "_select_scan", select_scan)
            page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, ""))
        elif entry == "mode":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_PROCESSING_MODE, "Int 1D"))
        elif entry == "detector":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_DETECTOR_MODE, "full"))
        else:
            page.close_workspace()
        if entry == "context":
            assert delegated == ["ACTIVE"] and page._background_owner.phase == "ACTIVE"
            return
        _assert_background_released(page._background_owner)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_a_detector_mode_change_keeps_an_int_background() -> None:
    """Full Raw or Thumbnail, also the raw pop-up's (thumbnail, then full, then thumbnail), changes only the raw
    image: an Int 2D background stays (before 1.2.18 it went). Int 1D's raw pop-up on a real browsed frame:
    test_int_1d_browse_background_follows_frames_through_idle_ticks."""
    qapp, page, _acquisition_context = _active_page_background()
    owner = page._background_owner
    try:
        key = owner.active_key
        for value, path in (("full", ()), ("thumbnail", ("popup",)), ("full", ("popup",)), ("thumbnail", ("popup",))):
            page._handle_shell_command(ShellCommand(ShellCommandKind.SET_DETECTOR_MODE, value, path))
            assert owner.phase == "ACTIVE" and owner.active_key == key, (value, path)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_a_1d_viewer_background_follows_into_another_file(tmp_path) -> None:
    """His pick "Any scan or file" in the 1D Viewer (LIVE-1218-VIEWER2D-BG): a real Set BG on one .xye file, then
    another opened: it is drawn minus the background (twice the first on the same axis: it shows the first)."""
    from tests.xdart.scattering.test_p2b1_viewer_context_page import _await_ready, _write_xye
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    x = np.linspace(1.0, 5.0, 9)
    first = _write_xye(tmp_path / "first.xye", x, 10.0 + x ** 2)
    second = _write_xye(tmp_path / "second.xye", x, 2.0 * (10.0 + x ** 2))
    chosen = [(str(first),)]
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="1D Viewer", project_root=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        viewer_1d_file_chooser=lambda _start: chosen.pop(0))

    def settle():
        _await_ready(page._context_controller)
        for _tick in range(3):
            page._drain_executor(); page._refresh_shell(); qapp.processEvents()

    def drawn():  # the curve the view mounted (it subtracts a 1-D background as it draws)
        items = page._shell.scientific.curve.listDataItems()
        return None if not items else np.asarray(items[-1].yData, np.float64)

    try:
        page._run_action(); settle()
        view = page._shell.scientific
        assert view.background.text() == "Set 1D BG"
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
        for _tick in range(50):
            page._drain_executor(); qapp.processEvents()
            if page._background_owner.phase == "ACTIVE":
                break
        page._refresh_shell(); qapp.processEvents()
        assert page._background_owner.phase == "ACTIVE", page._notice_text
        key = page._background_owner.active_key
        np.testing.assert_allclose(drawn(), 0.0, atol=1e-12)
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(second)))  # the browser's open
        settle()
        assert page._context_controller.viewer_1d_context.paths == (str(second),)
        assert page._background_owner.active_key == key and view.background.text() == "Clear 1D BG"
        np.testing.assert_allclose(drawn(), 10.0 + x ** 2)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_active_background_survives_idle_drain_ticks() -> None:
    """LIVE-1218-VIEWER2D-BG: every idle drain tick released an adopted background once the view had drawn it."""
    qapp, page, _acquisition_context = _active_page_background()
    owner = page._background_owner
    try:
        key = owner.active_key
        assert owner.phase == "ACTIVE" and key is not None
        for _tick in range(5):
            page._drain_executor(); qapp.processEvents(); page._refresh_shell()
            assert owner.phase == "ACTIVE" and owner.active_key == key
            assert page._shell.scientific.background.text() == "Clear 2D BG"
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_a_background_stays_when_the_frames_it_was_set_on_go() -> None:
    """Since 1.2.18 no idle tick releases a display background, also once the frames it was set on are gone (the
    acquisition released): it waits for the next frame on its grid, until Clear BG (his pick "Any scan or file")."""
    qapp, page, _acquisition_context = _active_page_background()
    controller = page._context_controller
    try:
        key = page._background_owner.active_key
        assert page._shell.scientific.background.text() == "Clear 2D BG"
        assert controller.release_acquisition(controller.run_identity)
        for _tick in range(3):
            page._drain_executor(); qapp.processEvents(); page._refresh_shell()
        assert page._background_owner.active_key == key
        assert page._shell.scientific.background.text() == "Clear 2D BG"
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_a_mode_change_releases_a_promoted_background_before_first_paint() -> None:
    """Codex's review (49e0ac70): a background promoted and not yet painted is released at a boundary through the
    exact key the view expects; another key is refused. The boundary here is a processing-mode change (a scan
    boundary keeps a background since 1.2.18)."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    controller, view = page._context_controller, page._shell.scientific
    controller._runtime.adopt_acquisition(identity, acquisition)
    page._refresh_shell()
    try:
        view.background.click()
        operations = page._workspace_operations
        worker = operations._slot._worker
        assert worker is not None
        worker.join(3)
        assert not worker.is_alive()
        # The same ordering as the drain: adopt before advancing navigation
        # and before reconciling the scientific view.
        update = operations.poll(operations.current_identity)
        assert page._consume_background_update(update)
        owner = page._background_owner
        assert owner.phase == "ACTIVE" and view.background.text() == "Set 2D BG"
        wrong_key = (*owner.active_key, "another background")
        assert not view.release_display_background(wrong_key).released
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PROCESSING_MODE, "Int 1D"))
        _assert_background_released(owner)
        assert view.background.text() == "Set 1D BG"
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize("mode", ("Int 2D", "Int 1D"))
def test_int_background_stays_on_ticks_once_its_frames_are_replaced(mode: str) -> None:
    """Since 1.2.18 an Int background whose frames were replaced without a frame or scan choice (here the
    acquisition released and another adopted; a seeded 1-D background on frames the page does not show) stays
    through idle ticks: it follows the frames shown now (his words: "the whole idea is to apply it across frames")."""
    if mode == "Int 2D":
        qapp, page, _acquisition_context = _active_page_background()
        owner = page._background_owner
        page._drain_executor(); qapp.processEvents(); page._refresh_shell()
        assert owner.phase == "ACTIVE"
        controller = page._context_controller
        assert controller.release_acquisition(controller.run_identity)
        identity, replacement = _acquisition()
        controller._runtime.adopt_acquisition(identity, replacement)
    else:
        qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        page = ScatteringWorkspace(
            intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
            sources=FilesystemSourceAdapter())
        page._edit_run_strip(ShellCommandKind.SET_PROCESSING_MODE, "Int 1D")
        owner = _seed_page_1d_background(page)
    try:
        key = owner.active_key
        assert owner.phase == "ACTIVE"
        for _tick in range(3):
            page._drain_executor(); qapp.processEvents(); page._refresh_shell()
        assert owner.phase == "ACTIVE" and owner.active_key == key
        assert page._shell.scientific.background.text().startswith("Clear ")
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize(("kind", "value"), (
    (ShellCommandKind.SET_PLOT_AXIS, "2Th"),
    (ShellCommandKind.SET_SLICE_ENABLED, True),
    (ShellCommandKind.SET_SLICE_CENTER, 5.0),
    (ShellCommandKind.SET_SLICE_WIDTH, 3.0),
))
def test_active_1d_background_stays_on_plot_axis_and_slice_edits(kind, value) -> None:
    """Since 1.2.18 a 1-D background is subtracted from each pattern drawn on its own native bins, so another plot
    axis of the same quantity (Q, 2θ) or a slice edit (Int 1D draws no slices) keeps it; a normalization edit still
    lets it go (test_active_1d_background_revokes_on_normalization_edit)."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    owner = _seed_page_1d_background(page)
    try:
        key = owner.active_key
        page._handle_shell_command(ShellCommand(kind, value))
        assert owner.phase == "ACTIVE" and owner.active_key == key
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_active_2d_background_revokes_on_an_image_axis_edit() -> None:
    """A 2-D background holds one image axis's map: another axis (or map) lets it go."""
    qapp, page, _acquisition_context = _active_page_background()
    owner = page._background_owner
    try:
        page._drain_executor(); qapp.processEvents(); page._refresh_shell()
        assert owner.phase == "ACTIVE"
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "2Th-Chi"))
        _assert_background_released(owner)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_an_image_axis_edit_cancels_a_staged_background() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    page._context_controller._runtime.adopt_acquisition(identity, acquisition)
    page._refresh_shell()
    try:
        page._shell.scientific.background.click()
        worker = page._workspace_operations._slot._worker
        assert worker is not None
        worker.join(3)
        assert page._background_owner.phase == "STAGED"
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "2Th-Chi"))
        page._drain_executor()
        _assert_background_released(page._background_owner)
        assert page._shell.scientific.background.text() == "Set 2D BG"
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_int_2d_background_follows_the_frames_of_its_scan() -> None:
    """His pick, 2026-11-21: "Int 2D follows frames too". A 2-D background stays across frame choices and is
    subtracted from every frame whose map has its axes, until Clear 2D BG."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition(frames=2)
    page._context_controller._runtime.adopt_acquisition(identity, acquisition)
    page._refresh_shell()
    controller, view, owner = page._context_controller, page._shell.scientific, page._background_owner

    def cake_of_selection():
        return controller.project_background_contributors()[0].view.intensity_2d

    def drawn():
        return np.asarray(view.cake.image.image, dtype=np.float64).T

    def settle():
        for _tick in range(3):
            page._drain_executor(); qapp.processEvents(); page._refresh_shell()

    try:
        frames = controller.navigation.frames
        assert len(frames) == 2
        first = controller.navigation.current
        second = next(frame for frame in frames if frame is not first)
        background = np.asarray(cake_of_selection(), dtype=np.float64)
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
        worker = page._workspace_operations._slot._worker
        assert worker is not None; worker.join(3); assert not worker.is_alive()
        page._drain_executor(); qapp.processEvents(); settle()
        assert owner.phase == "ACTIVE" and view.background.text() == "Clear 2D BG"
        np.testing.assert_allclose(drawn(), np.zeros_like(background))
        charged = owner.active_bytes

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME, frame=second, frames=(second,)))
        settle()
        assert controller.navigation.current is second
        assert owner.phase == "ACTIVE" and view.background.text() == "Clear 2D BG"
        np.testing.assert_allclose(drawn(), cake_of_selection() - background)
        assert not np.allclose(cake_of_selection(), background) and owner.active_bytes == charged
        assert "differ" not in view.background.toolTip()

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME, frame=first, frames=(first,)))
        settle()
        np.testing.assert_allclose(drawn(), np.zeros_like(background))

        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
        settle()
        _assert_background_released(owner)
        assert view.background.text() == "Set 2D BG"
        np.testing.assert_allclose(drawn(), background)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_2d_background_draws_a_map_on_other_axes_or_shape_as_it_is() -> None:
    """A 2-D background fits a map of its shape whose presented axes equal those of the frame it was set on; any
    other map is drawn as it is, the tip saying why, with no refusal (his design word)."""
    shell = make_shell_projection(frame_count=3, selected_index=2, heavy_indices=(2,), plot_mode="Single")
    heavy = shell.scientific.heavy; assert heavy is not None
    contributors = ((heavy.cake, heavy.cake_y.values, heavy.cake_x.values),)
    plan = DisplayBackgroundPlan(
        "integrated_2d", ("image",), (heavy.cake.shape,),
        ((heavy.cake_y.values.shape, heavy.cake_x.values.shape),),
        ((f"{heavy.cake_y.label}\0{heavy.cake_y.unit}", f"{heavy.cake_x.label}\0{heavy.cake_x.unit}"),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = owner.reserve(
        plan, contributors, stamp=OperationContextStamp(0, "context", 1),
        active_key=("context", 1, "integrated_2d", "Int 2D"),
        projection_keys=(id(heavy.frame),), projection_indices=(0,))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, contributors)
    state = replace(shell.scientific, processing_mode="Int 2D")
    frames = shell.navigation.frames

    def shown(new_heavy):
        return apply_presentation_background(replace(state, heavy=new_heavy), owner.projection(),
                                             owner.frame_values)

    own = shown(heavy)
    np.testing.assert_allclose(own.heavy.cake, np.zeros_like(heavy.cake), atol=1e-12)
    assert shown(heavy).heavy.cake is own.heavy.cake  # the same frame: the same array
    same_image_other_grid = replace(
        heavy, cake_x=replace(heavy.cake_x, values=heavy.cake_x.values * 2.0))
    skipped = shown(same_image_other_grid)
    assert skipped.heavy.cake is heavy.cake and "map axes differ" in skipped.background_note
    same_grid = shown(replace(heavy, frame=frames[0], cake=heavy.cake + 5.0))
    np.testing.assert_allclose(same_grid.heavy.cake, np.full(heavy.cake.shape, 5.0))
    assert same_grid.background_set and not same_grid.background_note
    other_axes = replace(heavy, frame=frames[1], cake=heavy.cake + 7.0,
                         cake_x=replace(heavy.cake_x, values=heavy.cake_x.values * 2.0))
    skipped = shown(other_axes)
    assert skipped.heavy.cake is other_axes.cake and skipped.background_set
    assert "map axes differ" in skipped.background_note
    other_shape = shown(replace(heavy, frame=frames[1], cake=np.ones((1, heavy.cake.shape[1])),
                                cake_y=replace(heavy.cake_y, values=heavy.cake_y.values[:1])))
    assert "does not fit this frame" in other_shape.background_note
    assert owner.active_bytes == receipt.retained_bytes + heavy.cake.size * 8


def test_following_background_matches_the_image_not_a_reused_frame_id() -> None:
    """A following background outlives the frames it was drawn on, so a frame's id can be reused by another frame:
    the values follow the image drawn (held weakly), never a bare id, and a 2-D background takes its axes from its
    first draw after Set."""
    shell = make_shell_projection(frame_count=3, selected_index=2, heavy_indices=(2,), plot_mode="Single")
    heavy = shell.scientific.heavy; assert heavy is not None
    raw = np.asarray(heavy.raw, dtype=np.float64)
    plan = DisplayBackgroundPlan("raw", ("image",), (raw.shape,), ((),), ((),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = owner.reserve(
        plan, ((raw,),), stamp=OperationContextStamp(0, "context", 1),
        active_key=("context", 1, "raw", "2D Viewer"), projection_keys=(id(heavy.frame),),
        projection_indices=(0,))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, ((raw,),))
    key = owner.projection()[1][0][0]
    np.testing.assert_allclose(owner.frame_values(key, raw), np.zeros_like(raw))
    other = raw + 3.0  # another frame's image under the same (reused) id and shape
    np.testing.assert_allclose(owner.frame_values(key, other), np.full(raw.shape, 3.0))
    again = owner.frame_values(key, other)
    assert again is owner.frame_values(key, other)  # the same image keeps the same array


def test_active_1d_background_stays_on_a_frame_choice(monkeypatch) -> None:
    """Since 1.2.18 a frame choice keeps a 1-D background (his words: "If it is dropped on selection change what is
    the point of it at all?"); it is subtracted from each pattern drawn (test_trace_values_*)."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    page._edit_run_strip(ShellCommandKind.SET_PROCESSING_MODE, "Int 1D")
    owner = _seed_page_1d_background(page)
    frame = make_shell_projection(frame_count=2, selected_index=1).navigation.frames[0]
    monkeypatch.setattr(page, "_select_frames", lambda _command: None)
    try:
        key = owner.active_key
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_FRAME, frame=frame, frames=(frame,)))
        assert owner.phase == "ACTIVE" and owner.active_key == key
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_active_1d_background_revokes_on_normalization_edit() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    owner = _seed_page_1d_background(page)
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_NORM_CHANNEL, "monitor"))
        _assert_background_released(owner)
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize("viewer_owned", (False, True))
def test_clear_1d_releases_active_background_before_target_clear(
    monkeypatch, viewer_owned: bool,
) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    page._edit_run_strip(ShellCommandKind.SET_PROCESSING_MODE, "Int 1D")
    owner, order = _seed_page_1d_background(page), []
    release = page._release_display_background

    def observed_release():
        order.append(("release", owner.phase))
        return release()

    monkeypatch.setattr(page, "_release_display_background", observed_release)
    if viewer_owned:
        monkeypatch.setattr(
            type(page._context_controller), "viewer_1d_owned",
            property(lambda _self: True),
        )
        monkeypatch.setattr(
            page, "_clear_viewer_1d_renderer",
            lambda *, close=False: order.append(("clear", owner.phase, close)) or True,
        )
    try:
        page._handle_shell_command(ShellCommand(ShellCommandKind.CLEAR_1D))
        _assert_background_released(owner)
        assert order[0] == ("release", "ACTIVE")
        if viewer_owned:
            assert order[1] == ("clear", "RELEASED", True)
        assert page._shell.scientific.background.text() == "Set 1D BG"
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize("domain", ("raw", "integrated_1d", "integrated_2d"))
def test_projection_substitutes_only_exact_domain_targets(domain: str) -> None:
    shell = make_shell_projection(frame_count=3, selected_index=2,
                                  heavy_indices=(2,), plot_mode="Overlay")
    before = shell.scientific
    assert before.heavy is not None
    if domain == "integrated_1d":
        rows = tuple((id(trace.frame), np.full(trace.intensity.shape, index + 10.0))
                     for index, trace in enumerate(before.traces))
    else:
        source = before.heavy.raw if domain == "raw" else before.heavy.cake
        rows = ((id(before.heavy.frame), np.full(source.shape, 42.0)),)
    after = apply_presentation_background(
        replace(before, processing_mode="Int 1D" if domain == "integrated_1d"
                else "Int 2D"), (domain, rows, ("active", domain)))
    assert after.background_set
    if domain == "raw":
        assert after.heavy.raw is rows[0][1] and after.heavy.cake is before.heavy.cake
        assert after.traces == before.traces
    elif domain == "integrated_2d":
        assert after.heavy.cake is rows[0][1] and after.heavy.raw is before.heavy.raw
        assert after.traces == before.traces
    else:
        assert all(trace.intensity is row[1]
                   for trace, row in zip(after.traces, rows, strict=True))
        assert after.heavy is before.heavy


@pytest.mark.parametrize("plot_mode", ("Single", "Overlay", "Waterfall"))
def test_1d_renderer_uses_reused_d_with_baseline_role_parity(plot_mode: str) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=4, selected_index=3,
                                  heavy_indices=(3,), plot_mode=plot_mode)
    state = replace(
        shell.scientific, processing_mode="Int 1D",
        plot_options=replace(shell.scientific.plot_options, overlay_offset=0.0))
    selected = tuple(trace for trace in state.traces
                     if any(trace.frame is frame for frame in shell.navigation.selected))
    contributors = tuple((trace.intensity, trace.axis.values) for trace in selected)
    plan = DisplayBackgroundPlan(
        "integrated_1d", tuple(f"trace-{index}" for index in range(len(selected))),
        tuple(trace.intensity.shape for trace in selected),
        tuple((trace.axis.values.shape,) for trace in selected),
        tuple((f"{trace.axis.label}\0{trace.axis.unit}",) for trace in selected))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    key = ("context", 1, "integrated_1d", plot_mode)
    reservation = owner.reserve(
        plan, contributors, stamp=OperationContextStamp(0, "context", 1),
        active_key=key, projection_keys=tuple(id(trace.frame) for trace in selected),
        projection_indices=tuple(range(len(selected))))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, tuple((*item, 1.0) for item in contributors))
    projection = owner.projection(); assert projection is not None
    d_ids = tuple(id(value) for _frame, value in projection[1])
    expected = tuple(value for _frame, value in projection[1])
    view = ScientificView()
    try:
        view.reconcile(state, shell.navigation, completed=4, total=4, detail="Ready")
        off_roles = _renderer_roles(view)
        raw_before = view.raw.canvas.raw_image.copy()
        cake_before = view.cake.canvas.raw_image.copy()
        view.expect_display_background(key)
        active = apply_presentation_background(state, projection)
        view.reconcile(active, shell.navigation, completed=4, total=4, detail="Ready")
        assert _renderer_roles(view) == off_roles
        if plot_mode == "Waterfall":
            np.testing.assert_allclose(view.waterfall.image.image.T,
                                       np.stack(expected), equal_nan=True)
        else:
            rendered = view.curve.listDataItems()
            assert len(rendered) == len(expected)
            for item, value in zip(rendered, expected, strict=True):
                np.testing.assert_allclose(item.getData()[1], value, equal_nan=True)
        view.reconcile(active, shell.navigation, completed=4, total=4, detail="Ready")
        assert tuple(id(value) for _frame, value in owner.projection()[1]) == d_ids
        np.testing.assert_array_equal(view.raw.canvas.raw_image, raw_before)
        np.testing.assert_array_equal(view.cake.canvas.raw_image, cake_before)
        renderer_receipt = view.release_display_background(key)
        assert owner.release(renderer_receipt)
        _assert_background_released(owner)
        view.reconcile(state, shell.navigation, completed=4, total=4, detail="Ready")
        assert _renderer_roles(view) == off_roles
    finally:
        view.close(); qapp.processEvents()


def _active_1d_owner(background, axis, *, unit="q\0A^-1", channel=""):
    plan = DisplayBackgroundPlan("integrated_1d", ("background",), (background.shape,), ((axis.shape,),),
                                 ((unit,),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = owner.reserve(
        plan, ((background, axis),), stamp=OperationContextStamp(0, "context", 1),
        active_key=("context", 1, plan.domain, "Int 1D", "integrated"), projection_keys=(11,),
        projection_indices=(0,), norm_channel=channel)
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, ((background, axis, 1.0),))
    return owner


def test_trace_values_follow_every_curve_drawn() -> None:
    """LIVE-1218-VIEWER2D-BG (his words: "the whole idea is to apply it across frames"): a 1-D background gives each
    curve drawn its drawn intensity minus the background; on another grid the background is taken at the curve's
    axis (NaN outside its range), on the same bins to display precision directly; another quantity or unit is a
    note, the curve drawn as it is; the same curve drawn again gets the same array, another curve under a reused
    id its own values."""
    axis = np.arange(4.0)
    background = np.array([1.0, 2.0, 3.0, 4.0])
    owner = _active_1d_owner(background, axis)
    pattern = np.array([10.0, 20.0, 30.0, 40.0])
    drawn = owner.trace_values(pattern, axis, "q\0A^-1")
    np.testing.assert_allclose(drawn, pattern - background)
    assert owner.trace_values(pattern, axis, "q\0A^-1") is drawn
    assert not drawn.flags.writeable
    note = owner.trace_values(pattern, axis, "2th\0deg")
    assert type(note) is str and "q (A^-1)" in note and "2th (deg)" in note
    shifted = np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(owner.trace_values(pattern, shifted, "q\0A^-1"),
                               [8.5, 17.5, 26.5, np.nan], equal_nan=True)
    rounded = (axis + 1e-8).astype(np.float32).astype(np.float64)  # the same bins stored otherwise
    np.testing.assert_allclose(owner.trace_values(pattern.copy(), rounded, "q\0A^-1"), pattern - background)
    other = pattern + 1.0
    np.testing.assert_allclose(owner.trace_values(other, axis, "q\0A^-1"), other - background)
    assert owner.frame_values(11, pattern) is None  # a 1-D background follows curves, not images


def test_follow_trace_takes_the_curves_own_basis_and_channel() -> None:
    """The view's follow corrects a curve from what it carries (its native axis, quantity and unit, and channel),
    so a kept curve whose frame is gone is corrected as the current one; a curve without a basis, or normalized by
    another channel than the background was set under, is a note (his pick "Each by its own monitor": the
    background is of contributors each divided by its own monitor, so a curve under another channel does not fit)."""
    axis = np.arange(4.0)
    owner = _active_1d_owner(np.array([1.0, 2.0, 3.0, 4.0]), axis, channel="I0")
    frame = make_shell_projection(frame_count=1, selected_index=0).navigation.frames[0]
    shown = TraceProjection(frame, AxisProjection(axis * 2.0, "2θ", "deg"), np.array([5.0, 6.0, 7.0, 8.0]),
                            background_basis=(axis, "q\0A^-1", "I0"))
    np.testing.assert_allclose(owner.follow_trace(shown), [4.0, 4.0, 4.0, 4.0])
    assert "no native axis" in owner.follow_trace(replace(shown, background_basis=None))
    note = owner.follow_trace(replace(shown, background_basis=(axis, "q\0A^-1", "")))
    assert "normalized by I0" in note and "not normalized" in note
    assert owner.release(DisplayBackgroundRendererReleaseReceipt(owner.active_key, True))
    assert owner.follow_trace(shown) is None


def test_following_a_dense_grid_shift_interpolates_instead_of_matching_bin_indices() -> None:
    axis = 1.0 + 2e-7 * np.arange(4)
    shifted = axis + 2e-7
    background = np.array([1.0, 10.0, 50.0, 100.0])
    owner = _active_1d_owner(background, axis)
    pattern = np.full(4, 100.0)
    np.testing.assert_allclose(
        owner.trace_values(pattern, shifted, "q\0A^-1"),
        pattern - np.interp(shifted, axis, background, left=np.nan, right=np.nan))


def test_a_single_following_pattern_must_fit_the_background_grant() -> None:
    axis = np.arange(1024.0)
    owner = _active_1d_owner(np.zeros_like(axis), axis)
    active = owner.active_key
    base = owner.active_bytes
    owner._capacity = base + axis.nbytes - 1
    result = owner.trace_values(np.ones_like(axis), axis, "q\0A^-1")
    assert type(result) is str and "allowance" in result
    assert owner.active_key == active and owner.active_bytes == base


def test_trace_values_charge_the_grant_and_go_with_the_background() -> None:
    """The patterns' values are charged to the owner's one grant and dropped, oldest first, past it; release
    drops them all. A raw or 2-D background gives none."""
    axis = np.arange(1024.0)
    owner = _active_1d_owner(np.zeros(1024), axis)
    base = owner.active_bytes
    patterns = [np.full(1024, float(index)) for index in range(4)]
    for pattern in patterns:
        owner.trace_values(pattern, axis, "q\0A^-1")
    assert owner.active_bytes == base + 4 * 1024 * 8
    owner._capacity = base + 2 * 1024 * 8  # a tight grant: only the two newest stay held
    owner.trace_values(np.ones(1024), axis, "q\0A^-1")
    assert owner.active_bytes <= owner._capacity and len(owner._followed) == 2
    owner._capacity = 536_870_912
    assert owner.release(DisplayBackgroundRendererReleaseReceipt(owner.active_key, True))
    _assert_background_released(owner)
    assert owner._followed == {} and owner.trace_values(patterns[0], axis, "q\0A^-1") is None


def test_trace_values_let_go_of_patterns_no_longer_drawn() -> None:
    """A long Run with a background set draws a new pattern per frame: the values of a pattern that is gone are
    dropped at the next draw, so the owner holds only the patterns still alive, and the same pattern drawn again
    on another axis array replaces its own values without double counting."""
    import gc
    axis = np.arange(1024.0)
    owner = _active_1d_owner(np.zeros(1024), axis)
    base = owner.active_bytes
    for index in range(50):
        owner.trace_values(np.full(1024, float(index)), axis, "q\0A^-1")  # each gone once drawn
    gc.collect()
    kept = np.ones(1024)
    owner.trace_values(kept, axis, "q\0A^-1")
    assert owner.active_bytes == base + 1024 * 8 and len(owner._followed) == 1
    owner.trace_values(kept, axis[::-1].copy()[::-1].copy(), "q\0A^-1")  # the same pattern on a new axis array
    assert owner.active_bytes == base + 1024 * 8 and len(owner._followed) == 1


def test_a_trace_carries_its_native_basis_and_channel() -> None:
    """trace_projection gives a native curve its background basis: its native axis, quantity and unit, and the one
    effective channel it was divided by (E6-NORM-N2's one resolution); a curve without a native row has none."""
    from xdart.gui.tabs.scattering.scientific_axes import trace_projection
    controller, _lifecycle, _executor, _loader, _acquisition_context = _running_controller()
    frame = controller.navigation.current; assert frame is not None
    view = replace(_view(1, 3.0), metadata_numeric={"monitor": 2.0})
    payload = StandardDisplayPayload(0, frame, "monitored", view)
    trace = trace_projection(payload, requested_axis=view.axis_1d.label, allow_cake=False, slice_enabled=False,
                             slice_center=0.0, slice_width=1.0, norm_channel="monitor")
    assert trace is not None
    np.testing.assert_allclose(trace.intensity, view.intensity_1d / 2.0)
    axis, unit, channel = trace.background_basis
    assert axis is view.axis_1d.values and unit == f"{view.axis_1d.label}\0{view.axis_1d.unit}"
    assert channel == "monitor"
    plain = trace_projection(payload, requested_axis=view.axis_1d.label, allow_cake=False, slice_enabled=False,
                             slice_center=0.0, slice_width=1.0)
    assert plain.background_basis[2] == ""


@pytest.mark.parametrize("plot_mode", ("Overlay", "Average", "Sum", "Waterfall"))
def test_the_view_subtracts_a_1d_background_from_every_curve_it_draws(plot_mode: str) -> None:
    """Codex's event-2 HOLD: the view subtracts a 1-D background at its one rows point, after the history merges,
    so every curve it draws is corrected before Overlay offsets, the Average and Sum folds and the Waterfall image;
    the history keeps the curves as they are, and the release brings them back."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=4, selected_index=3, heavy_indices=(3,), plot_mode=plot_mode)
    traces = tuple(replace(trace, background_basis=(trace.axis.values, f"{trace.axis.label}\0{trace.axis.unit}", ""))
                   for trace in shell.scientific.traces)
    state = replace(shell.scientific, processing_mode="Int 1D", traces=traces,
                    plot_options=replace(shell.scientific.plot_options, overlay_offset=0.0))
    axis = traces[0].axis
    background = np.asarray(traces[0].intensity, dtype=np.float64) * 0.5
    owner = _active_1d_owner(background, np.asarray(axis.values, dtype=np.float64),
                             unit=f"{axis.label}\0{axis.unit}")
    corrected = [np.asarray(trace.intensity, np.float64) - background for trace in traces
                 if any(trace.frame is frame for frame in shell.navigation.selected)]
    view = ScientificView()
    try:
        view.expect_display_background(owner.active_key)
        view.follow_display_background(owner.follow_trace)
        active = apply_presentation_background(state, owner.projection(), view_follows_traces=True)
        assert active.traces is state.traces and active.background_set
        view.reconcile(active, shell.navigation, completed=4, total=4, detail="Ready")
        if plot_mode == "Waterfall":
            np.testing.assert_allclose(view.waterfall.image.image.T, np.stack(corrected), rtol=1e-5, equal_nan=True)
        elif plot_mode in {"Average", "Sum"}:
            (item,) = view.curve.listDataItems()
            fold = np.nanmean if plot_mode == "Average" else np.nansum
            np.testing.assert_allclose(item.getData()[1], fold(np.stack(corrected), axis=0), rtol=1e-5)
        else:
            drawn = [item.getData()[1] for item in view.curve.listDataItems()]
            assert len(drawn) == len(corrected)
            for got, want in zip(drawn, corrected, strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-5, equal_nan=True)
        assert all(trace in traces for trace in view.trace_history_projections)  # kept as they are
        receipt = view.release_display_background(owner.active_key)
        assert receipt.released and owner.release(receipt)
        view.reconcile(replace(state, background_set=False), shell.navigation, completed=4, total=4, detail="Ready")
        if plot_mode == "Overlay":
            drawn = [item.getData()[1] for item in view.curve.listDataItems()]
            selected = [t for t in traces if any(t.frame is frame for frame in shell.navigation.selected)]
            for got, trace in zip(drawn, selected, strict=True):
                np.testing.assert_allclose(got, trace.intensity, rtol=1e-6)
    finally:
        view.close(); qapp.processEvents()


def test_the_detached_budget_counts_a_curves_native_basis_once() -> None:
    """A kept curve on a converted axis holds its native axis too (its background basis): the Browse comparison
    budget counts that root, once, and not again when the basis is the drawn axis itself."""
    from xdart.gui.tabs.scattering.scientific_view import _detached_root_bytes
    frame = make_shell_projection(frame_count=1, selected_index=0).navigation.frames[0]
    native = np.arange(64.0)
    shown = TraceProjection(frame, AxisProjection(native * 2.0, "2θ", "deg"), np.ones(64))
    plain = _detached_root_bytes((shown,))
    assert _detached_root_bytes((replace(shown, background_basis=(native, "q\0A^-1", "")),)) == plain + native.nbytes
    same = replace(shown, background_basis=(shown.axis.values, "2θ\0deg", ""))
    assert _detached_root_bytes((same,)) == plain


def test_the_view_keeps_the_background_refusal_note_on_an_idle_redraw() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=1, selected_index=0, plot_mode="Single")
    trace, = shell.scientific.traces
    trace = replace(trace, background_basis=(trace.axis.values, "q_oop\0A^-1", ""))
    state = replace(shell.scientific, processing_mode="Int 1D", traces=(trace,))
    owner = _active_1d_owner(trace.intensity, trace.axis.values, unit="q_ip\0A^-1")
    view = ScientificView()
    try:
        view.expect_display_background(owner.active_key)
        view.follow_display_background(owner.follow_trace)
        active = apply_presentation_background(state, owner.projection(), view_follows_traces=True)
        for _ in range(2):
            view.reconcile(active, shell.navigation, completed=1, total=1, detail="Ready")
            assert "q_oop (A^-1)" in view.background.toolTip()
    finally:
        view.close(); qapp.processEvents()


def test_the_view_releases_followed_angular_support_inputs_without_losing_history() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=1, selected_index=0)
    trace, = shell.scientific.traces
    trace = replace(trace, background_basis=(trace.axis.values, "q\0A^-1", ""))
    state = replace(shell.scientific, processing_mode="Int 1D", traces=(trace,))
    owner = _active_1d_owner(trace.intensity, trace.axis.values)
    view = ScientificView()
    try:
        view.expect_display_background(owner.active_key)
        view.follow_display_background(owner.follow_trace)
        active = apply_presentation_background(state, owner.projection(), view_follows_traces=True)
        view.reconcile(active, shell.navigation, completed=1, total=1, detail="Ready")
        followed = weakref.ref(owner.follow_trace(trace))
        assert view.trace_history_projections == (trace,)
        assert owner.release(view.release_display_background(owner.active_key))
        assert view.trace_history_projections == (trace,)
        assert followed() is None
    finally:
        view.close(); qapp.processEvents()


def test_the_view_only_corrects_waterfall_rows_it_draws() -> None:
    from xdart.gui.tabs.scattering.scientific_view import MAX_WATERFALL_DISPLAY_ROWS

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    count = MAX_WATERFALL_DISPLAY_ROWS + 2
    shell = make_shell_projection(frame_count=count, selected_index=count - 1, plot_mode="Waterfall")
    traces = tuple(replace(trace, background_basis=(trace.axis.values, "q\0A^-1", ""))
                   for trace in shell.scientific.traces)
    state = replace(shell.scientific, processing_mode="Int 1D", traces=traces)
    owner = _active_1d_owner(traces[0].intensity, traces[0].axis.values)
    view = ScientificView()
    try:
        view.expect_display_background(owner.active_key)
        view.follow_display_background(owner.follow_trace)
        active = apply_presentation_background(state, owner.projection(), view_follows_traces=True)
        view.reconcile(active, shell.navigation, completed=count, total=count, detail="Ready")
        assert view.waterfall.image.image.shape[1] == MAX_WATERFALL_DISPLAY_ROWS
        assert len(view.trace_history_projections) == count
        assert len(owner._followed) == MAX_WATERFALL_DISPLAY_ROWS
    finally:
        view.close(); qapp.processEvents()


def test_differing_grid_float32_contributors_are_taken_as_float64() -> None:
    """A Browse selection's detached patterns are float32 display copies: on two grids the background (an
    interpolated mean, which runs in float64) copies them as float64, charged at 8 bytes a value, instead of
    refusing the Set (his pick "In 1.2.18" for the two-frame Browse Set BG)."""
    first_axis, second_axis = np.arange(4.0), np.arange(4.0) + 0.5
    first = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    second = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    plan = DisplayBackgroundPlan("integrated_1d", ("a", "b"), ((4,), (4,)), (((4,),), ((4,),)),
                                 (("q\0A^-1",), ("q\0A^-1",)), True)
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = _reserve(owner, plan, ((first, first_axis.astype(np.float32)), (second, second_axis)),
                           keys=(11, 12), indices=(0, 1))
    assert reservation is not None
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, ((np.asarray(first, np.float64), first_axis, 1.0),
                                   (np.asarray(second, np.float64), second_axis, 1.0)))
    values = owner._result.values
    # the mean on the first's grid: the second interpolated there (NaN where it does not reach)
    np.testing.assert_allclose(values, [1.0, 2.75, 3.75, 4.75], equal_nan=True)


@pytest.mark.parametrize("mode", ("Average", "Sum"))
def test_an_average_of_curves_on_two_grids_folds_both_during_a_run(mode: str) -> None:
    """Since 1.2.18 Average and Sum keep the comparison (his pick "In 1.2.18"); during a Run (acquisition selected)
    two selected curves on different grids of one axis must still both reach the fold, on the common grid, as before,
    never one left out by the Overlay's grid admission."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=2, selected_index=1, plot_mode=mode)
    first, second = shell.scientific.traces[:2]
    axis = first.axis
    shifted = replace(second, axis=AxisProjection(np.asarray(axis.values) + 0.37 * float(np.diff(axis.values)[0]),
                                                  axis.label, axis.unit))
    state = replace(shell.scientific, processing_mode="Int 1D", traces=(first, shifted), acquisition_selected=True)
    view = ScientificView()
    try:
        view.reconcile(state, shell.navigation, completed=2, total=2, detail="Ready")
        (item,) = view.curve.listDataItems()
        x, y = item.getData()
        assert x.size > axis.values.size  # the common grid of both, not either one's alone
        reached_only_first = (x < float(np.min(shifted.axis.values)))
        inside = (x > float(np.min(shifted.axis.values))) & (x < float(np.max(axis.values)))
        assert inside.any() and np.isfinite(y[inside]).all()
        assert not np.allclose(y[inside], np.interp(x[inside], axis.values, first.intensity) * (2 if mode == "Sum" else 1))
        del reached_only_first
    finally:
        view.close(); qapp.processEvents()


def test_1d_background_keeps_exact_normalization_and_fails_closed_without_it() -> None:
    """His pick, 2026-11-22, "Each by its own monitor" (the question's formula I_frame/monitor_frame -
    <I_bg/monitor_bg>): the contributors are each divided by their own monitor before the mean, and a frame is
    drawn as its own normalized values minus that (before 1.2.18: (I - B) / m with B in raw counts)."""
    axis = np.arange(3.0)
    background = np.array([2.0, 4.0, 6.0])
    source = np.array([10.0, 20.0, 30.0])
    plan = DisplayBackgroundPlan(
        "integrated_1d", ("background",), ((3,),), (((3,),),),
        (("q\0A^-1",),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = _reserve(owner, plan, ((background, axis),), keys=(11,), indices=(0,))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, ((source, axis, 2.0),))
    displayed = owner.projection()[1][0][1]
    np.testing.assert_allclose(displayed, source / 2.0 - background)

    refused = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = _reserve(refused, plan, ((background, axis),), keys=(11,), indices=(0,))
    receipt = refused.run_and_stage(reservation, lambda: False)
    assert not refused.promote(receipt, ((source, axis, None),))

    controller, _lifecycle, _executor, _loader, _acquisition_context = _running_controller()
    frame = controller.navigation.current; assert frame is not None
    view = replace(_view(1, 1.0), metadata_numeric={"monitor": 2.0})
    monitored = StandardDisplayPayload(0, frame, "monitored", view)
    prepared = prepare_background_plan(
        (monitored,), "integrated_1d", monitored.frame_key,
        contributor_count=1, norm_channel="monitor")
    assert prepared is not None
    assert prepared[3][0][2] == 2.0
    assert prepared[1][0][0] is view.intensity_1d
    plan, contributors, keys, targets, _facts = prepared
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reserve_kwargs = dict(stamp=OperationContextStamp(0), active_key=("background",),
                          projection_keys=keys, projection_shapes=tuple(target[0].shape for target in targets),
                          norm_channel="monitor", contributor_divisors=(targets[0][2],))
    reservation = owner.reserve(plan, contributors, **reserve_kwargs)
    assert reservation is not None
    np.testing.assert_allclose(owner._contributors[0][0], view.intensity_1d / 2.0)
    too_small = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    too_small._capacity = owner.reserved_bytes - 1
    assert too_small.reserve(plan, contributors, **reserve_kwargs) is None
    assert prepared[4] == ("monitor", ((id(monitored.frame_key), (2,), ((2,),),
                                        ("q\0A^-1",), 2.0),))
    assert prepare_background_plan(
        (replace(monitored, view=_view(1, 1.0)),), "integrated_1d", frame,
        contributor_count=1, norm_channel="monitor") is None


def test_background_censuses_pin_only_target_separately_from_contributors() -> None:
    controller, _lifecycle, _executor, _loader, acquisition = _running_controller()
    first = controller.navigation.current; assert first is not None
    display = acquisition.publication_store
    delta = display.append_navigation("run.a", "/out/a.nxs", 2)
    second = delta.appended
    view = _view(2, 4.0)
    display.put_payload(StandardDisplayPayload(
        0, second, "Standard · run.a · frame 2", view))
    assert controller.accept_navigation(
        delta, plot_mode="Overlay", follow_latest=False)
    assert controller.select_navigation(first, (first,))
    payloads = controller.project_background_contributors(
        (SlicePin(second, "Q", 0.5, 1.0),))
    assert tuple(payload.frame_key for payload in payloads) == (first, second)
    payloads = tuple(
        replace(payload, view=_view(index, float(index + 2)))
        for index, payload in enumerate(payloads, 1))
    prepared = prepare_background_plan(
        payloads, "integrated_1d", first, contributor_count=1)
    assert prepared is not None
    plan, contributors, keys, targets, target_facts = prepared
    assert len(plan.contributor_ids) == len(contributors) == 1
    assert keys == (id(first), id(second))
    assert len(targets) == len(target_facts[1]) == 2


@pytest.mark.parametrize("domain", ("raw", "integrated_2d"))
def test_image_renderer_uses_d_and_preserves_the_non_target_pane(domain: str) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=3, selected_index=2,
                                  heavy_indices=(2,), plot_mode="Single")
    heavy = shell.scientific.heavy; assert heavy is not None
    if domain == "raw":
        state = replace(shell.scientific, processing_mode="2D Viewer")
        contributors = ((heavy.raw,),); axis_shapes, units = ((),), ((),)
    else:
        state = replace(shell.scientific, processing_mode="Int 2D")
        contributors = ((heavy.cake, heavy.cake_y.values, heavy.cake_x.values),)
        axis_shapes = (((heavy.cake_y.values.shape, heavy.cake_x.values.shape)),)
        units = ((f"{heavy.cake_y.label}\0{heavy.cake_y.unit}",
                  f"{heavy.cake_x.label}\0{heavy.cake_x.unit}"),)
    value = contributors[0][0]
    plan = DisplayBackgroundPlan(
        domain, ("image",), (value.shape,), axis_shapes, units)
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    key = ("context", 1, domain, state.processing_mode)
    reservation = owner.reserve(
        plan, contributors, stamp=OperationContextStamp(0, "context", 1),
        active_key=key, projection_keys=(id(heavy.frame),), projection_indices=(0,))
    receipt = owner.run_and_stage(reservation, lambda: False)
    assert owner.promote(receipt, contributors)
    projection = owner.projection(); assert projection is not None
    d_value = projection[1][0][1]; d_id = id(d_value)
    view = ScientificView()
    try:
        view.reconcile(state, shell.navigation, completed=3, total=3, detail="Ready")
        off_roles = _renderer_roles(view)
        opposite = view.cake if domain == "raw" else view.raw
        opposite_before = opposite.canvas.raw_image.copy()
        view.expect_display_background(key)
        active = apply_presentation_background(state, projection)
        view.reconcile(active, shell.navigation, completed=3, total=3, detail="Ready")
        assert _renderer_roles(view) == off_roles
        target = view.raw if domain == "raw" else view.cake
        expected = d_value.T[:, ::-1] if domain == "raw" else d_value.T
        np.testing.assert_allclose(target.image.image, expected, equal_nan=True)
        np.testing.assert_array_equal(opposite.canvas.raw_image, opposite_before)
        view.reconcile(active, shell.navigation, completed=3, total=3, detail="Ready")
        assert id(owner.projection()[1][0][1]) == d_id
        renderer_receipt = view.release_display_background(key)
        assert domain != "raw" or view._viewer_2d_payload is None
        assert owner.release(renderer_receipt); _assert_background_released(owner)
        np.testing.assert_array_equal(opposite.canvas.raw_image, opposite_before)
        view.reconcile(state, shell.navigation, completed=3, total=3, detail="Ready")
        assert _renderer_roles(view) == off_roles
    finally:
        view.close(); qapp.processEvents()


def test_domain_labels_raw_popup_and_exact_target_renderer_release() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = make_shell_projection(frame_count=3, selected_index=2,
                                  heavy_indices=(2,), plot_mode="Overlay")
    view = ScientificView()
    labels = {"Int 1D": "Set 1D BG", "1D Viewer": "Set 1D BG",
              "Int 2D": "Set 2D BG", "2D Viewer": "Set Raw BG"}
    assert {mode: ScatteringWorkspace._background_domain(mode) for mode in labels} == {
        "Int 1D": "integrated_1d", "Int 2D": "integrated_2d",
        "1D Viewer": "integrated_1d", "2D Viewer": "raw"}
    assert ScatteringWorkspace._background_domain("Int 1D Raw popup") is None
    try:
        for mode, label in labels.items():
            state = replace(shell.scientific, processing_mode=mode,
                            background_set=False)
            view.reconcile(state, shell.navigation, completed=3, total=3,
                           detail="Ready")
            assert view.background.text() == label and not view.background.isHidden()
        state = replace(shell.scientific, processing_mode="Int 1D",
                        background_set=True)
        key = ("context", 1, "integrated_1d", "Int 1D")
        view.expect_display_background(key)
        view.reconcile(state, shell.navigation, completed=3, total=3, detail="Ready")
        view._ensure_raw_popup()
        assert view.background.text() == "Clear 1D BG"
        cake_before = view.cake.image.image.copy()
        wrong = view.release_display_background((*key[:-1], "other"))
        assert not wrong.released and view.curve.listDataItems()
        receipt = view.release_display_background(key)
        assert receipt.released and not view.curve.listDataItems()
        np.testing.assert_array_equal(view.cake.image.image, cake_before)
    finally:
        if view.raw_popup_dialog is not None: view.raw_popup_dialog.close()
        view.close(); qapp.processEvents()


def test_real_page_one_click_one_worker_adopts_and_clear_preserves_source(monkeypatch) -> None:
    from xdart.gui.tabs.scattering import presentation_background as owner_module

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    page._context_controller._runtime.adopt_acquisition(identity, acquisition)
    page._refresh_shell()
    payload = page._context_controller.project_background_contributors()[0]
    source = payload.view.intensity_2d; source_before = source.copy()
    display = acquisition.publication_store
    artifact = next(iter(display.artifacts.values()))
    store_before = (
        id(display), id(artifact),
        tuple((key, id(value)) for key, value in artifact.records.snapshot().items()),
        tuple((key, id(value)) for key, value in artifact.publications.snapshot().items()),
        tuple((id(key), id(value)) for key, value in display.payloads.items()),
        tuple(id(key) for key in display.catalog_snapshot().entries),
    )
    intent_before = page._intents.snapshot()
    calls, runner = [], owner_module.run_display_background
    def observed(*args, **kwargs):
        calls.append(args[0]); return runner(*args, **kwargs)
    monkeypatch.setattr(owner_module, "run_display_background", observed)
    try:
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
        worker = page._workspace_operations._slot._worker
        assert worker is not None; worker.join(3); assert not worker.is_alive()
        page._drain_executor(); qapp.processEvents()
        projection = page._background_owner.projection()
        assert len(calls) == 1 and projection is not None
        assert page._background_owner.phase == "ACTIVE"
        assert page._intents.snapshot() == intent_before
        assert page._shell.scientific.background.text() == "Clear 2D BG"
        np.testing.assert_array_equal(source, source_before)
        displayed = projection[1][0][1]
        np.testing.assert_allclose(displayed, np.zeros_like(displayed))
        active_key, displayed_id = page._background_owner.active_key, id(displayed)
        page._edit_run_strip(ShellCommandKind.SET_CORES, 2)
        assert page._background_owner.active_key == active_key
        assert id(page._background_owner.projection()[1][0][1]) == displayed_id
        intent_after_unrelated_edit = page._intents.snapshot()
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
        assert page._background_owner.phase == "RELEASED"
        assert page._shell.scientific.background.text() == "Set 2D BG"
        np.testing.assert_array_equal(source, source_before)
        assert page._intents.snapshot() == intent_after_unrelated_edit
        assert store_before == (
            id(display), id(artifact),
            tuple((key, id(value)) for key, value in artifact.records.snapshot().items()),
            tuple((key, id(value)) for key, value in artifact.publications.snapshot().items()),
            tuple((id(key), id(value)) for key, value in display.payloads.items()),
            tuple(id(key) for key in display.catalog_snapshot().entries),
        )
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_ast_has_one_public_runner_edge_and_no_hot_path_or_peer_worker() -> None:
    root = Path(__file__).parents[3]
    public = (root / "src/xrd_tools/reduction/background.py").read_text()
    owner = (root / "src/xdart/gui/tabs/scattering/presentation_background.py").read_text()
    page = (root / "src/xdart/gui/tabs/scattering/page.py").read_text()
    assert public.count("def run_display_background(") == 1
    assert owner.count("run_display_background(") == 1
    public_tree = ast.parse(public)
    imports = {
        alias.name for node in ast.walk(public_tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(public_tree)
         if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("PyQt", "PySide", "qtpy", "xdart"))
                   for name in imports)
    assert all(name not in public + owner for name in (
        "NexusSink", "NexusRecordWriter", "write_batch", "PublicationStore",
        "FrameRecordStore", "integrate_1d", "integrate_2d"))
    assert all(name not in owner for name in ("Thread(", "Queue(", "Timer(", "cache"))
    assert page.count("PresentationBackgroundOwner(capacity_bytes=536_870_912)") == 1
    # one contributor authority (_background_contributors, since 1.2.18 with the Browse detached fallback), used by
    # the Set and its adoption alone, never by a refresh
    assert page.count("project_background_contributors(") == 1
    assert page.count("ScatteringWorkspace._background_contributors(self,") == 2
    tree = ast.parse(page)
    refresh = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_refresh_shell")
    assert "project_background_contributors" not in ast.unparse(refresh)
    assert "_background_contributors" not in ast.unparse(refresh)
    owner_tree = ast.parse(owner)
    promote = next(node for node in ast.walk(owner_tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "promote")
    assert "source.reshape" not in ast.unparse(promote)
