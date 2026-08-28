from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.controls_editing import reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import (
    INT_1D_AXIS,
    INT_2D_AXIS,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.core import Axis, FrameView, TwoDKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _gi_intent() -> RunIntent:
    intent = RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/raw/eiger"),
            suffixes=(".h5",),
        ),
        project_root="/raw/eiger",
        save_path="/processed",
        poni_file="/calibration/detector.poni",
        bai_1d_args={"unit": "qip_A^-1"},
        bai_2d_args={"unit": "qip_A^-1"},
    )
    intent.gi.enabled = True
    intent.gi.mode_1d = "q_ip"
    intent.gi.mode_2d = "qip_qoop"
    return intent


def _gi_projection(
    *,
    mode_1d: str,
    mode_2d: str,
    axis_1d: Axis,
    axis_2d_x: Axis,
    axis_2d_y: Axis,
    kind: TwoDKind,
    share_axis: bool = True,
    plot_axis: str = "Q",
    processing_mode: str = "Int 2D",
    slice_enabled: bool = False,
    slice_center: float = 0.0,
    slice_width: float = 10.0,
):
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    x_size = axis_2d_x.values.size
    y_size = axis_2d_y.values.size
    payload = StandardDisplayPayload(
        0,
        frame,
        "GI frame",
        FrameView(
            frame.local_frame_label,
            axis_1d=axis_1d,
            intensity_1d=np.linspace(1.0, 2.0, axis_1d.values.size),
            axis_2d_x=axis_2d_x,
            axis_2d_y=axis_2d_y,
            intensity_2d=np.arange(
                x_size * y_size,
                dtype=float,
            ).reshape(y_size, x_size),
            two_d_kind=kind,
            raw=np.ones((3, 4), dtype=float),
        ),
        measurement_mode="GI",
        gi_incidence_motor="halpha",
        gi_resolved_motor="halpha",
        gi_mode_1d=mode_1d,
        gi_mode_2d=mode_2d,
        wavelength_m=1.0e-10,
    )
    state = build_scientific_projection(
        (payload,),
        base.navigation,
        frozenset({frame}),
        ScientificPreferences(
            plot_axis=plot_axis,
            image_axis="Q-Chi",
            plot_mode="Single",
            share_axis=share_axis,
            slice_enabled=slice_enabled,
            slice_center=slice_center,
            slice_width=slice_width,
        ),
        "",
        processing_mode=processing_mode,
    )
    return state, base.navigation


def _combo_items(combo) -> tuple[tuple[str, object], ...]:
    return tuple(
        (combo.itemText(index), combo.itemData(index))
        for index in range(combo.count())
    )


def _reconcile(state, navigation):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    view.reconcile(
        state,
        navigation,
        completed=1,
        total=1,
        detail="Ready",
    )
    app.processEvents()
    return app, view


def test_gi_controls_use_the_production_axis_inventory_and_semantics() -> None:
    store = RunIntentStore(_gi_intent())
    snapshot = store.snapshot()
    state = project_controls(snapshot, None, RunPhase.IDLE)
    fields = {
        field.path: field
        for field in state.fields
    }

    assert fields[INT_1D_AXIS].choices == (
        "Q",
        "Qip",
        "Qoop",
        "Exit",
        "χGI",
    )
    assert fields[INT_1D_AXIS].value == "Qip"
    assert fields[INT_2D_AXIS].choices == (
        "Qip-Qoop",
        "Q-χ",
        "Exit",
    )
    assert fields[INT_2D_AXIS].value == "Qip-Qoop"

    one_d = reduce_control_edit(snapshot, INT_1D_AXIS, "Qoop")
    assert type(one_d) is RunIntent
    assert one_d.gi.mode_1d == "q_oop"
    assert one_d.bai_1d_args["unit"] == "qoop_A^-1"

    two_d = reduce_control_edit(snapshot, INT_2D_AXIS, "Exit")
    assert type(two_d) is RunIntent
    assert two_d.gi.mode_2d == "exit_angles"


def test_gi_qip_qoop_uses_native_toolbar_vocabulary_and_shares_qip() -> None:
    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, navigation = _gi_projection(
        mode_1d="q_ip",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
    )

    assert state.measurement_mode == "GI"
    assert state.gi_mode_1d == "q_ip"
    assert state.gi_mode_2d == "qip_qoop"
    assert state.plot_axis == "q_ip"
    assert state.image_axis == "qip_qoop"
    assert state.traces[0].axis.unit == "qip_A^-1"
    assert state.heavy is not None
    assert state.heavy.cake_x is not None
    assert state.heavy.cake_x.unit == "qip_A^-1"

    _app, view = _reconcile(state, navigation)
    try:
        assert _combo_items(view.plot_axis) == (
            ("Qᵢₚ (Å⁻¹)", "q_ip"),
            ("Qₒₒₚ (Å⁻¹)", "q_oop"),
        )
        assert _combo_items(view.image_axis) == (
            ("Qᵢₚ-Qₒₒₚ", "qip_qoop"),
        )
        assert not view.image_axis.isEnabled()
        assert view.share_axis.isEnabled()
        assert view.share_axis.isChecked()
        assert view._share_link_on
    finally:
        view.close()


def test_retained_hydration_keeps_last_actually_rendered_gi_cake_identity() -> None:
    """Share Axis follows the visible cake, not a newer unpainted projection."""

    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, navigation = _gi_projection(
        mode_1d="q_ip",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
    )
    _app, view = _reconcile(state, navigation)
    try:
        assert view.rendered_image_axis == "qip_qoop"
        retained = replace(
            state,
            heavy=None,
            traces=(),
            measurement_mode="Standard",
            gi_mode_1d="",
            gi_mode_2d="",
            image_axis="Q-Chi",
            plot_axis="Q",
            retain_display=True,
        )

        view.reconcile(
            retained,
            navigation,
            completed=1,
            total=1,
            detail="Hydrating",
        )

        assert view.rendered_image_axis == "qip_qoop"
        assert view.cake.canvas.displayed_image.size
    finally:
        view.close()


def test_gi_qoop_native_repoints_to_cake_derived_qip_for_share() -> None:
    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, navigation = _gi_projection(
        mode_1d="q_oop",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_oop", "qoop_A^-1", values=qoop),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
    )

    assert state.plot_axis == "q_ip"
    assert state.image_axis == "qip_qoop"
    assert state.traces[0].axis.unit == "qip_A^-1"
    np.testing.assert_allclose(
        state.traces[0].intensity,
        np.arange(35, dtype=float).reshape(5, 7).mean(axis=0),
    )
    _app, view = _reconcile(state, navigation)
    try:
        assert _combo_items(view.plot_axis) == (
            ("Qₒₒₚ (Å⁻¹)", "q_oop"),
            ("Qᵢₚ (Å⁻¹)", "q_ip"),
        )
        assert view.plot_axis.currentData() == "q_ip"
        assert view.share_axis.isEnabled()
        assert view.share_axis.isChecked()
        assert view._share_link_on
    finally:
        view.close()


def test_gi_q_total_q_chi_keeps_only_native_cake_and_valid_plot_choices() -> None:
    q = np.linspace(0.1, 4.0, 7)
    chi = np.linspace(-90.0, 90.0, 5)
    state, navigation = _gi_projection(
        mode_1d="q_total",
        mode_2d="q_chi",
        axis_1d=Axis("Q", "q_A^-1", values=q),
        axis_2d_x=Axis("Q", "q_A^-1", values=q),
        axis_2d_y=Axis("chi", "chi_deg", values=chi),
        kind=TwoDKind.Q_CHI,
    )

    assert state.plot_axis == "Q"
    assert state.image_axis == "q_chi"
    _app, view = _reconcile(state, navigation)
    try:
        assert _combo_items(view.plot_axis) == (
            ("Q (Å⁻¹)", "Q"),
            ("2θ (°)", "2theta"),
            ("χ (°)", "chi"),
        )
        assert _combo_items(view.image_axis) == (
            ("Q-χ", "q_chi"),
        )
        assert not view.image_axis.isEnabled()
        assert view.share_axis.isEnabled()
        assert view.share_axis.isChecked()
    finally:
        view.close()


@pytest.mark.parametrize(
    (
        "requested",
        "slice_center",
        "slice_width",
        "expected_axis",
        "expected_intensity",
        "expected_suffix",
    ),
    (
        (
            "q_ip",
            2.0,
            0.01,
            "qip_A^-1",
            np.arange(35, dtype=float).reshape(5, 7)[2, :],
            " · Qᵢₚ@Qₒₒₚ=2.00±0.01",
        ),
        (
            "q_oop",
            0.0,
            0.01,
            "qoop_A^-1",
            np.arange(35, dtype=float).reshape(5, 7)[:, 3],
            " · Qₒₒₚ@Qᵢₚ=0.00±0.01",
        ),
    ),
)
def test_gi_cake_projection_slices_over_the_complementary_axis(
    requested: str,
    slice_center: float,
    slice_width: float,
    expected_axis: str,
    expected_intensity: np.ndarray,
    expected_suffix: str,
) -> None:
    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, _navigation = _gi_projection(
        mode_1d="q_oop",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_oop", "qoop_A^-1", values=qoop),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
        share_axis=False,
        plot_axis=requested,
        slice_enabled=True,
        slice_center=slice_center,
        slice_width=slice_width,
    )

    assert state.plot_axis == requested
    assert state.traces[0].axis.unit == expected_axis
    assert state.traces[0].title == f"scan-a_1{expected_suffix}"
    np.testing.assert_array_equal(
        state.traces[0].intensity,
        expected_intensity,
    )


def test_empty_complementary_axis_slice_publishes_no_trace() -> None:
    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, _navigation = _gi_projection(
        mode_1d="q_ip",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
        share_axis=False,
        plot_axis="q_ip",
        slice_enabled=True,
        slice_center=99.0,
        slice_width=0.01,
    )

    assert state.traces == ()


def test_gi_int_1d_menu_excludes_every_cake_only_axis() -> None:
    qip = np.linspace(-3.0, 3.0, 7)
    qoop = np.linspace(0.0, 4.0, 5)
    state, navigation = _gi_projection(
        mode_1d="q_oop",
        mode_2d="qip_qoop",
        axis_1d=Axis("Q_oop", "qoop_A^-1", values=qoop),
        axis_2d_x=Axis("Q_ip", "qip_A^-1", values=qip),
        axis_2d_y=Axis("Q_oop", "qoop_A^-1", values=qoop),
        kind=TwoDKind.QIP_QOOP,
        share_axis=False,
        plot_axis="q_ip",
        processing_mode="Int 1D",
    )

    assert state.plot_axis == "q_oop"
    assert state.traces[0].axis.unit == "qoop_A^-1"
    _app, view = _reconcile(state, navigation)
    try:
        assert _combo_items(view.plot_axis) == (
            ("Qₒₒₚ (Å⁻¹)", "q_oop"),
        )
        assert view.share_axis.isHidden()
    finally:
        view.close()


@pytest.mark.parametrize(
    (
        "mode_1d",
        "mode_2d",
        "axis_1d",
        "axis_2d_x",
        "axis_2d_y",
        "kind",
        "expected_plot_axis",
        "expected_unit",
        "expected_choices",
    ),
    (
        (
            "chi_gi",
            "q_chi",
            Axis("chi_GI", "chigi_deg", values=np.linspace(-2.0, 2.0, 5)),
            Axis("Q", "q_A^-1", values=np.linspace(0.1, 4.0, 7)),
            Axis("chi", "chi_deg", values=np.linspace(-90.0, 90.0, 5)),
            TwoDKind.Q_CHI,
            "Q",
            "q_A^-1",
            (
                ("χGI (°)", "chi_gi"),
                ("Q (Å⁻¹)", "Q"),
                ("χ (°)", "chi"),
            ),
        ),
        (
            "exit_angle",
            "exit_angles",
            Axis(
                "Exit angle",
                "exit_angle_deg",
                values=np.linspace(-2.0, 2.0, 7),
            ),
            Axis(
                "Exit angle",
                "exit_angle_deg",
                values=np.linspace(-2.0, 2.0, 7),
            ),
            Axis(
                "Exit angle",
                "exit_angle_deg",
                values=np.linspace(-1.0, 1.0, 5),
            ),
            TwoDKind.EXIT_ANGLES,
            "exit_angle",
            "exit_angle_deg",
            (("Exit angle (°)", "exit_angle"),),
        ),
    ),
)
def test_gi_share_repoints_by_each_rendered_cake_x_identity(
    mode_1d: str,
    mode_2d: str,
    axis_1d: Axis,
    axis_2d_x: Axis,
    axis_2d_y: Axis,
    kind: TwoDKind,
    expected_plot_axis: str,
    expected_unit: str,
    expected_choices: tuple[tuple[str, str], ...],
) -> None:
    state, navigation = _gi_projection(
        mode_1d=mode_1d,
        mode_2d=mode_2d,
        axis_1d=axis_1d,
        axis_2d_x=axis_2d_x,
        axis_2d_y=axis_2d_y,
        kind=kind,
    )

    assert state.plot_axis == expected_plot_axis
    assert state.traces[0].axis.unit == expected_unit
    _app, view = _reconcile(state, navigation)
    try:
        assert _combo_items(view.plot_axis) == expected_choices
        assert view.share_axis.isChecked()
        assert view._share_link_on
    finally:
        view.close()
