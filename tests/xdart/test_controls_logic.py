from xdart.gui.tabs.static_scan.controls_logic import (
    AnalysisTool,
    ControlState,
    MeasMode,
    ResultCaps,
    SourceCaps,
    Tool,
    GeomState,
    build_analysis_launchers,
    build_control_profile,
)


def _launcher(specs, tool):
    return next(s for s in specs if s.tool == tool)


def test_analysis_launchers_keep_live_peak_fit_openable_before_data():
    specs = build_analysis_launchers(ResultCaps(has_1d=False))
    peak = _launcher(specs, AnalysisTool.PEAK_FIT)
    phase = _launcher(specs, AnalysisTool.PHASE_FIT)

    assert peak.enabled
    assert peak.live_capable
    assert peak.batch_capable
    assert "Waiting" in peak.reason
    assert phase.enabled


def test_analysis_launchers_can_gate_known_missing_optional_dependencies():
    specs = build_analysis_launchers(
        ResultCaps(has_1d=True, available_optional_deps=frozenset()))

    assert not _launcher(specs, AnalysisTool.PEAK_FIT).enabled
    assert not _launcher(specs, AnalysisTool.PHASE_FIT).enabled
    assert _launcher(specs, AnalysisTool.SCAN_PLOT).enabled


def test_future_analysis_tools_are_present_but_gated_by_data():
    specs = build_analysis_launchers(
        ResultCaps(has_1d=True, has_psi_metadata=False, raw_reachable=False))

    assert not _launcher(specs, AnalysisTool.ROI_STATS).enabled
    assert not _launcher(specs, AnalysisTool.SIN2PSI).enabled
    assert not _launcher(specs, AnalysisTool.TEXTURE).production_ready


def test_gi_stitch_and_rsm_controls_are_ready_but_blocked_until_gates():
    common = dict(
        source_caps=SourceCaps(
            has_frames=True, has_raw=True, raw_reachable=True,
            has_metadata=True, has_motors=True, has_energy=True,
            has_geometry=True),
        geom=GeomState(
            calibrated=True, energy_known=True, gi_enabled=True,
            sample_orientation_known=True, ub_known=True),
    )

    gi_stitch = build_control_profile(
        ControlState(tool=Tool.STITCH, mode=MeasMode.GI, **common))
    assert not gi_stitch.run_enabled
    assert "GI stitching awaits real-data gate." in gi_stitch.run_blockers

    rsm = build_control_profile(
        ControlState(tool=Tool.RSM, mode=MeasMode.STANDARD, **common))
    assert not rsm.run_enabled
    assert "RSM GUI awaits real-data gate." in rsm.run_blockers

    rsm_ready = build_control_profile(
        ControlState(
            tool=Tool.RSM, mode=MeasMode.STANDARD,
            real_data_gates=frozenset({"rsm_real_data"}), **common))
    assert rsm_ready.run_enabled


def test_viewer_modes_do_not_offer_run_even_without_blockers():
    profile = build_control_profile(
        ControlState(tool=Tool.IMAGE_VIEWER, source_caps=SourceCaps(has_frames=True)))
    assert not profile.run_enabled
    assert not profile.show_processing_card
