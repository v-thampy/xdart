from __future__ import annotations

import numpy as np
import pytest

from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_projection import (
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ProgressProjection,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.core import FrameView, IntegrationResult1D
from xrd_tools.session.run_configuration import RunIntent

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.mark.parametrize("phase", tuple(RunPhase))
def test_image_member_name_owns_live_status_and_plot_title(
    phase: RunPhase,
) -> None:
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
        f"Standard · {frame.source_scan} · frame {frame.local_frame_label}",
        FrameView(
            frame.local_frame_label,
            raw=np.ones((2, 3)),
            source_path="/raw/Combi4_scan_0016.tif",
            source_frame_index=0,
        ),
        "running",
    )

    projected = ContextProjection().build_shell(
        revision=2,
        controls=base.controls,
        controls_readiness=base.controls_readiness,
        phase=phase,
        intent=RunIntent(),
        contexts=(),
        selection=None,
        navigation=base.navigation,
        payloads=(payload,),
        resident_frames=frozenset({frame}),
        progress=ProgressProjection(1, 1, ""),
        preferences=ScientificPreferences(plot_mode="Single"),
        browser_directory="",
        date_sorted=False,
        auto_last=True,
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        notice="",
    )

    assert projected.scientific.status == "Combi4_scan_0016.tif"
    assert projected.scientific.title == "Combi4_scan_0016.tif"
    assert frame.artifact not in projected.scientific.status


@pytest.mark.parametrize(
    ("source_path", "source_index", "expected"),
    [
        (
            "/raw/eiger_w2s4_1_eta_0p118_scan002_master.h5",
            0,
            "eiger_w2s4_1_eta_0p118_scan002_master.h5 · frame 1",
        ),
        (
            "/raw/eiger_w2s4_1_eta_0p118_scan002_master.h5",
            3,
            "eiger_w2s4_1_eta_0p118_scan002_master.h5 · frame 4",
        ),
        (
            "/raw/bluesky_17_2_00090.nxs",
            12,
            "bluesky_17_2_00090.nxs · frame 13",
        ),
        ("/processed/scan_00090.nexus", 12, "scan_00090.nexus · frame 13"),
    ],
)
def test_container_member_name_includes_exact_source_frame(
    source_path: str,
    source_index: int,
    expected: str,
) -> None:
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
        "synthetic title must not win",
        FrameView(
            frame.local_frame_label,
            raw=np.ones((2, 3)),
            source_path=source_path,
            source_frame_index=source_index,
        ),
        "running",
    )

    projected = ContextProjection().build_shell(
        revision=2,
        controls=base.controls,
        controls_readiness=base.controls_readiness,
        phase=RunPhase.IDLE,
        intent=RunIntent(),
        contexts=(),
        selection=None,
        navigation=base.navigation,
        payloads=(payload,),
        resident_frames=frozenset({frame}),
        progress=ProgressProjection(1, 1, ""),
        preferences=ScientificPreferences(plot_mode="Single"),
        browser_directory="",
        date_sorted=False,
        auto_last=True,
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        notice="",
    )

    assert projected.scientific.status == expected
    assert projected.scientific.title == expected


def test_container_trace_legend_uses_one_based_source_frame() -> None:
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
        "synthetic",
        FrameView.from_results(
            label=frame.local_frame_label,
            result_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.linspace(1.0, 2.0, 8),
                unit="q_A^-1",
            ),
            source_path="/raw/eiger_master.h5",
            source_frame_index=0,
        ),
        "ready",
    )
    navigation = FrameNavigationProjection(
        (frame,),
        frame,
        (frame,),
    )

    projected = build_scientific_projection(
        (payload,),
        navigation,
        frozenset({frame}),
        ScientificPreferences(plot_mode="Single"),
        "",
    )

    assert len(projected.traces) == 1
    assert projected.traces[0].title == f"{frame.source_scan}_1"


def test_live_status_notice_remains_higher_priority() -> None:
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    frame = base.navigation.current
    assert frame is not None
    display_payload = StandardDisplayPayload(
        0,
        frame,
        base.scientific.title,
        FrameView(
            frame.local_frame_label,
            raw=np.ones((2, 3)),
            source_path="/raw/member_0001.tif",
            source_frame_index=0,
        ),
        "running",
    )

    projected = ContextProjection().build_shell(
        revision=2,
        controls=base.controls,
        controls_readiness=base.controls_readiness,
        phase=RunPhase.IDLE,
        intent=RunIntent(),
        contexts=(),
        selection=None,
        navigation=base.navigation,
        payloads=(display_payload,),
        resident_frames=frozenset({frame}),
        progress=ProgressProjection(1, 1, ""),
        preferences=ScientificPreferences(plot_mode="Single"),
        browser_directory="",
        date_sorted=False,
        auto_last=True,
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        notice="Output cleanup remains pending.",
    )

    assert (
        projected.scientific.status
        == "Output cleanup remains pending."
    )
    assert projected.scientific.title == "member_0001.tif"
