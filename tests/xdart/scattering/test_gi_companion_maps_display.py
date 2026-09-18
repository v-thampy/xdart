"""GI-COMPANION-20260918 — which 2-D map the pane shows is a display choice.

A GI frame offers the maps it actually has: its primary, any other DIRECT map
stored with it, and -- for a q_ip–q_oop frame with no direct q–χ -- a q–χ map
re-binned for display.  Choosing one is pure presentation, the 1-D pane's
cake-derived axes (I–Q, I–χ) follow the shown map whatever the 1-D integration
unit was, and everything read off the re-binned map says so.

Drives the real projection builder and the real pane widget.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.scientific_axes import (
    DERIVED_Q_CHI,
    available_gi_maps,
    present_gi_map,
)
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xrd_tools.core import FrameView, TwoDKind, axis_from_unit

from tests.xdart.scattering.e3_shell_support import make_shell_projection

QIP = np.linspace(-1.5, 1.5, 61)
QOOP = np.linspace(0.02, 1.8, 45)
Q = np.linspace(0.05, 2.2, 50)
CHI = np.linspace(-80.0, 80.0, 40)


def _cartesian_cake() -> np.ndarray:
    """A ring at |q| = 1.0 Å⁻¹, with a block of unmeasured bins."""
    qq = np.hypot(QIP[None, :], QOOP[:, None])
    cake = 5.0 + 100.0 * np.exp(-0.5 * ((qq - 1.0) / 0.05) ** 2)
    cake[:6, :10] = np.nan
    return cake


def _view(mode_1d: str = "q_ip") -> FrameView:
    axis_1d = (
        axis_from_unit("qip_A^-1", QIP) if mode_1d == "q_ip"
        else axis_from_unit("qtot_A^-1", Q)
    )
    return FrameView(
        0,
        axis_1d=axis_1d,
        intensity_1d=np.linspace(1.0, 2.0, axis_1d.values.size),
        axis_2d_x=axis_from_unit("qip_A^-1", QIP),
        axis_2d_y=axis_from_unit("qoop_A^-1", QOOP),
        intensity_2d=_cartesian_cake(),
        two_d_kind=TwoDKind.QIP_QOOP,
        raw=np.ones((3, 4), dtype=float),
    )


def _direct_q_chi() -> FrameView:
    return FrameView(
        0,
        axis_2d_x=axis_from_unit("qtot_A^-1", Q),
        axis_2d_y=axis_from_unit("chigi_deg", CHI),
        intensity_2d=np.full((CHI.size, Q.size), 7.0),
        two_d_kind=TwoDKind.QTOT_CHIGI,
    )


def _state(*, image_axis: str, plot_axis: str = "q_ip", direct: bool = False,
           mode_1d: str = "q_ip", slice_enabled: bool = False):
    base = make_shell_projection(
        frame_count=1, selected_index=0, heavy_indices=(0,), plot_mode="Single",
    )
    frame = base.navigation.current
    payload = StandardDisplayPayload(
        0, frame, "GI frame",
        _relabel(_view(mode_1d), frame),
        measurement_mode="GI",
        gi_incidence_motor="th", gi_resolved_motor="th",
        gi_mode_1d=mode_1d, gi_mode_2d="qip_qoop",
        wavelength_m=1.0e-10,
        extra_views_2d=(
            {"q_chi": _relabel(_direct_q_chi(), frame)} if direct else {}
        ),
    )
    state = build_scientific_projection(
        (payload,), base.navigation, frozenset({frame}),
        ScientificPreferences(
            plot_axis=plot_axis, image_axis=image_axis, plot_mode="Single",
            share_axis=False, slice_enabled=slice_enabled,
            slice_center=0.0, slice_width=30.0,
        ),
        "", processing_mode="Int 2D",
    )
    return state, base.navigation, payload


def _relabel(view: FrameView, frame) -> FrameView:
    return replace(view, label=frame.local_frame_label)


def _items(combo) -> tuple[tuple[str, object], ...]:
    return tuple((combo.itemText(i), combo.itemData(i)) for i in range(combo.count()))


def _reconciled(state, navigation) -> ScientificView:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    view.reconcile(state, navigation, completed=1, total=1, detail="Ready")
    app.processEvents()
    return view


def test_a_plain_qip_qoop_frame_offers_its_map_and_a_derived_q_chi():
    state, navigation, payload = _state(image_axis="Q-Chi")

    assert available_gi_maps(payload) == ("qip_qoop", DERIVED_Q_CHI)
    assert state.gi_maps == ("qip_qoop", DERIVED_Q_CHI)
    # Nothing was chosen, so the primary map is shown untouched: no eager remap.
    assert state.image_axis == "qip_qoop" and not state.derived_2d
    assert state.heavy.cake is payload.view.intensity_2d

    view = _reconciled(state, navigation)
    try:
        assert _items(view.image_axis) == (
            ("Qᵢₚ-Qₒₒₚ", "qip_qoop"),
            ("Q-χ (derived)", DERIVED_Q_CHI),
        )
        assert view.image_axis.isEnabled()
        assert _items(view.plot_axis) == (
            ("Qᵢₚ (Å⁻¹)", "q_ip"),
            ("Qₒₒₚ (Å⁻¹)", "q_oop"),
        )
    finally:
        view.close()


def test_the_derived_map_is_labelled_wherever_it_is_shown_or_cut():
    state, navigation, payload = _state(image_axis=DERIVED_Q_CHI, plot_axis="Q")

    assert state.derived_2d and state.image_axis == DERIVED_Q_CHI
    assert state.gi_mode_2d == "q_chi"
    assert (state.heavy.cake_x.unit, state.heavy.cake_y.unit) == (
        "qtot_A^-1", "chigi_deg",
    )
    # The frame itself is untouched: presentation never edits the record.
    assert payload.view.two_d_kind is TwoDKind.QIP_QOOP and not payload.derived_2d
    # I–Q is read off the derived map although the 1-D integration was q_ip...
    (trace,) = state.traces
    assert trace.axis.unit == "qtot_A^-1"
    assert trace.title.endswith("derived Q-χ")
    # ...and the ring the Cartesian cake holds at |q| = 1.0 is where it peaks.
    assert abs(float(trace.axis.values[np.nanargmax(trace.intensity)]) - 1.0) < 0.06

    view = _reconciled(state, navigation)
    try:
        assert _items(view.plot_axis) == (
            ("Qᵢₚ (Å⁻¹)", "q_ip"),
            ("Q (Å⁻¹, derived)", "Q"),
            ("χGI (°, derived)", "chi_gi"),
        )
        assert view.image_axis.currentData() == DERIVED_Q_CHI
    finally:
        view.close()


def test_i_chi_is_available_from_the_derived_map():
    state, _navigation, _payload = _state(image_axis=DERIVED_Q_CHI, plot_axis="chi_gi")
    (trace,) = state.traces
    assert trace.axis.unit == "chigi_deg"
    assert trace.title.endswith("derived Q-χ")
    assert np.isfinite(trace.intensity).any()


def test_a_frame_with_a_direct_q_chi_offers_it_and_never_a_derived_one():
    state, navigation, payload = _state(image_axis="q_chi", plot_axis="chi_gi", direct=True)

    assert available_gi_maps(payload) == ("qip_qoop", "q_chi")
    assert state.image_axis == "q_chi" and not state.derived_2d
    assert np.all(state.heavy.cake == 7.0)
    (trace,) = state.traces
    assert trace.axis.unit == "chigi_deg" and "derived" not in trace.title
    np.testing.assert_allclose(trace.intensity, 7.0)

    view = _reconciled(state, navigation)
    try:
        assert _items(view.image_axis) == (
            ("Qᵢₚ-Qₒₒₚ", "qip_qoop"),
            ("Q-χ", "q_chi"),
        )
        assert _items(view.plot_axis) == (
            ("Qᵢₚ (Å⁻¹)", "q_ip"),
            ("Q (Å⁻¹)", "Q"),
            ("χGI (°)", "chi_gi"),
        )
    finally:
        view.close()


def test_a_native_q_result_is_not_relabelled_as_derived():
    """With a q_total 1-D result, plain I–Q is the direct integration."""
    state, _navigation, _payload = _state(
        image_axis=DERIVED_Q_CHI, plot_axis="Q", mode_1d="q_total",
    )
    (trace,) = state.traces
    assert "derived" not in trace.title
    np.testing.assert_allclose(trace.intensity, np.linspace(1.0, 2.0, Q.size))


def test_an_unavailable_preference_falls_back_to_the_primary_map():
    state, _navigation, payload = _state(image_axis="q_chi")
    assert state.image_axis == "qip_qoop" and not state.derived_2d
    assert present_gi_map(payload, "exit_angles") is payload


@pytest.mark.parametrize("mode_2d", ("q_chi", "exit_angles"))
def test_other_primary_modes_keep_their_single_locked_entry(mode_2d):
    base = make_shell_projection(
        frame_count=1, selected_index=0, heavy_indices=(0,), plot_mode="Single",
    )
    frame = base.navigation.current
    shown = _relabel(_direct_q_chi(), frame)
    payload = StandardDisplayPayload(
        0, frame, "GI frame", shown, measurement_mode="GI",
        gi_incidence_motor="th", gi_resolved_motor="th",
        gi_mode_1d="q_total", gi_mode_2d=mode_2d, wavelength_m=1.0e-10,
    )
    assert available_gi_maps(payload) == (mode_2d,)
    state = build_scientific_projection(
        (payload,), base.navigation, frozenset({frame}),
        ScientificPreferences(image_axis=DERIVED_Q_CHI, plot_mode="Single"),
        "", processing_mode="Int 2D",
    )
    assert not state.derived_2d and state.heavy.cake is shown.intensity_2d
    view = _reconciled(state, base.navigation)
    try:
        assert view.image_axis.count() == 1 and not view.image_axis.isEnabled()
    finally:
        view.close()
