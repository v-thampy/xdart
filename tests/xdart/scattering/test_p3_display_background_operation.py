"""Focused GUI contract for the P3 display-background operation."""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.context_projection import _apply_presentation_background
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
    FrameNavigationProjection, ShellCommand, ShellCommandKind, SlicePin,
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


def _active_page_background(mode="Int 2D"):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())
    identity, acquisition = _acquisition()
    page._context_controller._runtime.adopt_acquisition(identity, acquisition)
    if mode != "Int 2D":
        page._edit_run_strip(ShellCommandKind.SET_PROCESSING_MODE, mode)
    page._refresh_shell()
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_BACKGROUND))
    worker = page._operation_slot._worker
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
    active = _apply_presentation_background(state, owner.projection())
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
    worker = page._operation_slot._worker
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


@pytest.mark.parametrize("entry", ("selection", "context", "mode", "detector", "close"))
def test_active_background_revokes_at_every_page_boundary(monkeypatch, entry: str) -> None:
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
            monkeypatch.setattr(page, "_select_scan",
                                lambda _value: delegated.append(page._background_owner.phase))
            page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, ""))
        elif entry == "mode":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_PROCESSING_MODE, "Int 1D"))
        elif entry == "detector":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_DETECTOR_MODE, "full"))
        else:
            page.close_workspace()
        _assert_background_released(page._background_owner)
        if entry in {"selection", "context"}: assert delegated == ["RELEASED"]
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
    after = _apply_presentation_background(
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
        active = _apply_presentation_background(state, projection)
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


def test_1d_background_keeps_exact_normalization_and_fails_closed_without_it() -> None:
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
    np.testing.assert_allclose(displayed, (source - background) / 2.0)

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
        active = _apply_presentation_background(state, projection)
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
        worker = page._operation_slot._worker
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
    assert page.count("project_background_contributors(") == 2
    tree = ast.parse(page)
    refresh = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_refresh_shell")
    assert "project_background_contributors" not in ast.unparse(refresh)
    owner_tree = ast.parse(owner)
    promote = next(node for node in ast.walk(owner_tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "promote")
    assert "source.reshape" not in ast.unparse(promote)
