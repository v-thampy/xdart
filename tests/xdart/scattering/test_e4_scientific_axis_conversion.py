from __future__ import annotations

import numpy as np
import pytest

from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xrd_tools.core import Axis, FrameView, TwoDKind

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _project(
    *,
    unit: str,
    values: np.ndarray,
    requested: str,
    wavelength_m: float | None,
):
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    cake = np.arange(12, dtype=float).reshape(3, 4)
    payload = StandardDisplayPayload(
        0,
        frame,
        "frame",
        FrameView(
            frame.local_frame_label,
            axis_2d_x=Axis("radial", unit, values=values),
            axis_2d_y=Axis(
                "chi",
                "chi_deg",
                values=np.linspace(-90.0, 90.0, 3),
            ),
            intensity_2d=cake,
            two_d_kind=TwoDKind.Q_CHI,
            raw=np.ones((2, 2)),
        ),
        wavelength_m=wavelength_m,
    )
    return build_scientific_projection(
        (payload,),
        base.navigation,
        frozenset({frame}),
        ScientificPreferences(
            image_axis=requested,
            plot_mode="Single",
        ),
        "",
    )


def _project_trace(
    *,
    unit: str,
    values: np.ndarray,
    requested: str,
    wavelength_m: float | None,
):
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    payload = StandardDisplayPayload(
        0,
        frame,
        "frame",
        FrameView(
            frame.local_frame_label,
            axis_1d=Axis("radial", unit, values=values),
            intensity_1d=np.linspace(1.0, 2.0, values.size),
            raw=np.ones((2, 2)),
        ),
        wavelength_m=wavelength_m,
    )
    return build_scientific_projection(
        (payload,),
        base.navigation,
        frozenset({frame}),
        ScientificPreferences(
            plot_axis=requested,
            plot_mode="Single",
        ),
        "",
    )


def test_display_only_q_cake_projects_to_two_theta_from_payload_wavelength():
    q = np.linspace(0.0, 5.0, 4)
    state = _project(
        unit="q_A^-1",
        values=q,
        requested="2Th-Chi",
        wavelength_m=1.0e-10,
    )

    assert state.image_axis == "2Th-Chi"
    assert state.heavy is not None
    assert state.heavy.cake_x is not None
    assert state.heavy.cake_x.unit == "2th_deg"
    np.testing.assert_allclose(
        state.heavy.cake_x.values,
        2.0 * np.rad2deg(np.arcsin(q / (4.0 * np.pi))),
    )


def test_missing_wavelength_keeps_native_q_axis_and_truthful_selector():
    q = np.linspace(0.0, 5.0, 4)
    state = _project(
        unit="q_A^-1",
        values=q,
        requested="2Th-Chi",
        wavelength_m=None,
    )

    assert state.image_axis == "Q-Chi"
    assert state.heavy is not None
    assert state.heavy.cake_x is not None
    assert state.heavy.cake_x.unit == "q_A^-1"
    np.testing.assert_array_equal(state.heavy.cake_x.values, q)


def test_display_only_two_theta_cake_projects_back_to_q():
    two_theta = np.linspace(0.0, 45.0, 4)
    state = _project(
        unit="2th_deg",
        values=two_theta,
        requested="Q-Chi",
        wavelength_m=1.0e-10,
    )

    assert state.image_axis == "Q-Chi"
    assert state.heavy is not None
    assert state.heavy.cake_x is not None
    assert state.heavy.cake_x.unit == "q_A^-1"
    np.testing.assert_allclose(
        state.heavy.cake_x.values,
        4.0 * np.pi * np.sin(np.deg2rad(two_theta) / 2.0),
    )


def test_display_only_q_trace_projects_to_two_theta_from_payload_wavelength():
    q = np.linspace(0.0, 5.0, 4)
    state = _project_trace(
        unit="q_A^-1",
        values=q,
        requested="2theta",
        wavelength_m=1.0e-10,
    )

    assert state.plot_axis == "2theta"
    assert len(state.traces) == 1
    assert state.traces[0].axis.unit == "2th_deg"
    np.testing.assert_allclose(
        state.traces[0].axis.values,
        2.0 * np.rad2deg(np.arcsin(q / (4.0 * np.pi))),
    )


def test_missing_wavelength_refuses_trace_conversion_and_selector_is_truthful():
    q = np.linspace(0.0, 5.0, 4)
    state = _project_trace(
        unit="q_A^-1",
        values=q,
        requested="2theta",
        wavelength_m=None,
    )

    assert state.plot_axis == "Q"
    assert len(state.traces) == 1
    assert state.traces[0].axis.unit == "q_A^-1"
    np.testing.assert_array_equal(state.traces[0].axis.values, q)


def test_display_only_two_theta_trace_projects_back_to_q():
    two_theta = np.linspace(0.0, 45.0, 4)
    state = _project_trace(
        unit="2th_deg",
        values=two_theta,
        requested="Q",
        wavelength_m=1.0e-10,
    )

    assert state.plot_axis == "Q"
    assert len(state.traces) == 1
    assert state.traces[0].axis.unit == "q_A^-1"
    np.testing.assert_allclose(
        state.traces[0].axis.values,
        4.0 * np.pi * np.sin(np.deg2rad(two_theta) / 2.0),
    )


def test_standard_chi_plot_is_projected_from_the_cake_y_axis():
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    q = np.linspace(0.0, 3.0, 4)
    chi = np.linspace(-1.0, 1.0, 3)
    cake = np.arange(12, dtype=float).reshape(3, 4)
    payload = StandardDisplayPayload(
        0,
        frame,
        "frame",
        FrameView(
            frame.local_frame_label,
            axis_1d=Axis("Q", "q_A^-1", values=q),
            intensity_1d=np.linspace(1.0, 2.0, q.size),
            axis_2d_x=Axis("Q", "q_A^-1", values=q),
            axis_2d_y=Axis("chi", "chi_deg", values=chi),
            intensity_2d=cake,
            two_d_kind=TwoDKind.Q_CHI,
            raw=np.ones((2, 2)),
        ),
        wavelength_m=1.0e-10,
    )

    state = build_scientific_projection(
        (payload,),
        base.navigation,
        frozenset({frame}),
        ScientificPreferences(
            plot_axis="chi",
            plot_mode="Single",
        ),
        "",
    )

    assert state.plot_axis == "chi"
    assert state.traces[0].axis.unit == "chi_deg"
    np.testing.assert_array_equal(
        state.traces[0].intensity,
        cake.mean(axis=1),
    )


def test_share_uses_the_truthfully_rendered_radial_identity():
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    q = np.linspace(0.0, 3.0, 4)
    chi = np.linspace(-1.0, 1.0, 3)
    payload = StandardDisplayPayload(
        0,
        frame,
        "frame",
        FrameView(
            frame.local_frame_label,
            axis_1d=Axis("Q", "q_A^-1", values=q),
            intensity_1d=np.linspace(1.0, 2.0, q.size),
            axis_2d_x=Axis("Q", "q_A^-1", values=q),
            axis_2d_y=Axis("chi", "chi_deg", values=chi),
            intensity_2d=np.arange(12, dtype=float).reshape(3, 4),
            two_d_kind=TwoDKind.Q_CHI,
            raw=np.ones((2, 2)),
        ),
        wavelength_m=None,
    )

    state = build_scientific_projection(
        (payload,),
        base.navigation,
        frozenset({frame}),
        ScientificPreferences(
            image_axis="2Th-Chi",
            plot_axis="2theta",
            plot_mode="Single",
            share_axis=True,
        ),
        "",
    )

    assert state.image_axis == "Q-Chi"
    assert state.plot_axis == "Q"
    assert state.heavy is not None
    assert state.heavy.cake_x is not None
    assert state.heavy.cake_x.unit == "q_A^-1"
    assert state.traces[0].axis.unit == "q_A^-1"


@pytest.mark.parametrize("processing_mode", ["Int 1D", "Int 2D"])
@pytest.mark.parametrize("with_cake", [False, True])
def test_share_follows_only_an_int_2d_cake_that_is_drawn(with_cake, processing_mode):
    """Share Axis follows the map shown (HOSTED-REDS-20261017 F1): a native 2θ pattern keeps its 2θ with Share on
    unless Int 2D draws a complete cake, whose Q it then shares. Stitch keeps Share on for every result."""
    base = make_shell_projection(frame_count=1, selected_index=0, plot_mode="Single")
    frame = base.navigation.current
    two_theta = np.linspace(5.0, 40.0, 4)
    cake = dict(axis_2d_x=Axis("Q", "q_A^-1", values=np.linspace(0.5, 3.0, 4)),
                axis_2d_y=Axis("chi", "chi_deg", values=np.linspace(-90.0, 90.0, 3)),
                intensity_2d=np.arange(12, dtype=float).reshape(3, 4), two_d_kind=TwoDKind.Q_CHI)
    payload = StandardDisplayPayload(0, frame, "frame", FrameView(
        frame.local_frame_label, axis_1d=Axis("2theta", "2th_deg", values=two_theta),
        intensity_1d=np.linspace(1.0, 2.0, two_theta.size), thumbnail=np.ones((2, 2)),
        **(cake if with_cake else {})), wavelength_m=1.0e-10)
    preferences = ScientificPreferences(image_axis="Q-Chi", plot_axis="2theta", plot_mode="Single", share_axis=True)
    state = build_scientific_projection((payload,), base.navigation, frozenset({frame}), preferences, "",
                                        processing_mode=processing_mode)
    shared = with_cake and processing_mode == "Int 2D"
    assert state.heavy is not None and (state.heavy.cake is not None) is with_cake
    assert state.plot_axis == ("Q" if shared else "2theta")
    assert state.traces[0].axis.unit == ("q_A^-1" if shared else "2th_deg")
    assert state.share_axis, "the preference itself is never rewritten"

def test_int_1d_shows_the_pattern_with_a_slice_left_on_from_int_2d():
    """Int 1D has no cake, so a slice (Int 2D's, still on after the mode changed) cannot apply: the frame's
    I(q) shows, where the slice's rule (never the full row for a cut) left the plot empty."""
    base = make_shell_projection(frame_count=1, selected_index=0, plot_mode="Single")
    frame = base.navigation.current
    q = np.linspace(0.5, 5.0, 4)
    payload = StandardDisplayPayload(0, frame, "frame", FrameView(
        frame.local_frame_label, axis_1d=Axis("radial", "q_A^-1", values=q),
        intensity_1d=np.linspace(1.0, 2.0, q.size), raw=np.ones((2, 2))))
    preferences = ScientificPreferences(plot_axis="Q", plot_mode="Single", slice_enabled=True)
    state = build_scientific_projection((payload,), base.navigation, frozenset({frame}), preferences, "",
                                        processing_mode="Int 1D")
    (trace,) = state.traces
    np.testing.assert_array_equal(trace.intensity, payload.view.intensity_1d)
    # Int 2D keeps the rule: a slice of a frame without a cake shows nothing
    assert build_scientific_projection((payload,), base.navigation, frozenset({frame}), preferences, "",
                                       processing_mode="Int 2D").traces == ()
