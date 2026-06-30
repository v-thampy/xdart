from xdart.gui.tabs.static_scan.controls_logic import (
    AnalysisTool,
    BOUND_CONTROL_PATHS,
    ControlAction,
    ControlFieldKind,
    ControlState,
    FieldId,
    INTEGRATOR_BACKED_CONTROL_PATHS,
    INTEGRATOR_BACKED_CONTROL_SPECS,
    SectionId,
    StatusKind,
    MeasMode,
    ResultCap,
    ResultCaps,
    SourceCaps,
    Tool,
    GeomState,
    INTEGRATION_CONTROL_PATHS,
    build_field_statuses,
    build_analysis_launchers,
    build_bound_control_state,
    build_control_panel_state,
    build_control_profile,
    coerce_control_edit_value,
    run_blockers_from_fields,
    run_required_fields_for,
    tool_from_mode_text,
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


def test_analysis_launchers_carry_context_contract_metadata():
    specs = build_analysis_launchers(
        ResultCaps(has_1d=True, raw_reachable=True, has_scan_metadata=True))

    peak = _launcher(specs, AnalysisTool.PEAK_FIT)
    roi = _launcher(specs, AnalysisTool.ROI_STATS)
    scan_plot = _launcher(specs, AnalysisTool.SCAN_PLOT)

    assert peak.entry_point.endswith("peak_fit_dialog:PeakFitDialog")
    assert peak.required_caps == frozenset({ResultCap.HAS_1D})
    assert peak.optional_deps == frozenset({"fitting"})
    assert peak.singleton_key == AnalysisTool.PEAK_FIT.value
    assert roi.required_caps == frozenset({ResultCap.RAW_REACHABLE})
    assert roi.singleton_key == "roi_stats"
    assert scan_plot.required_caps == frozenset({ResultCap.SCAN_METADATA})


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
    assert gi_stitch.valid_modes == frozenset(MeasMode)
    assert gi_stitch.backend_required == "pyfai_hist"
    assert not gi_stitch.run_enabled
    assert "GI stitching awaits real-data gate." in gi_stitch.run_blockers

    gi_stitch_gate_passed_without_backend = build_control_profile(
        ControlState(
            tool=Tool.STITCH,
            mode=MeasMode.GI,
            real_data_gates=frozenset({"gi_stitch_real_data"}),
            **common,
        )
    )
    assert not gi_stitch_gate_passed_without_backend.run_enabled
    assert (
        "Select backend pyfai_hist."
        in gi_stitch_gate_passed_without_backend.run_blockers
    )

    rsm = build_control_profile(
        ControlState(tool=Tool.RSM, mode=MeasMode.STANDARD, **common))
    assert rsm.valid_modes == frozenset()
    assert rsm.backend_required is None
    assert not rsm.run_enabled
    assert "RSM GUI awaits real-data gate." in rsm.run_blockers

    rsm_ready = build_control_profile(
        ControlState(
            tool=Tool.RSM, mode=MeasMode.STANDARD,
            real_data_gates=frozenset({"rsm_real_data"}), **common))
    assert rsm_ready.run_enabled
    assert rsm_ready.can_run


def test_viewer_modes_do_not_offer_run_even_without_blockers():
    profile = build_control_profile(
        ControlState(tool=Tool.IMAGE_VIEWER, source_caps=SourceCaps(has_frames=True)))
    assert not profile.run_enabled
    assert not profile.show_processing_card


def test_tool_from_legacy_mode_text():
    assert tool_from_mode_text("Int 1D") == Tool.INT_1D
    assert tool_from_mode_text("Int 2D") == Tool.INT_2D
    assert tool_from_mode_text("Int 1D (XYE)") == Tool.INT_1D
    assert tool_from_mode_text("Image Viewer") == Tool.IMAGE_VIEWER
    assert tool_from_mode_text("XYE Viewer") == Tool.XYE_VIEWER
    assert tool_from_mode_text("NeXus Viewer") == Tool.NEXUS_VIEWER
    assert tool_from_mode_text("Stitch 2D") == Tool.STITCH
    assert tool_from_mode_text("RSM") == Tool.RSM


def test_build_field_statuses_tracks_source_geometry_and_results():
    fields = build_field_statuses(
        ControlState(
            tool=Tool.INT_2D,
            project_root="/data",
            source_label="/data/scan001.nxs",
            save_path="/data/out",
            frame_count=3,
            processing_mode="Int 2D",
            source_caps=SourceCaps(
                has_frames=True, has_raw=True, raw_reachable=True,
                has_metadata=True, has_motors=True, has_energy=True,
                has_geometry=True),
            result_caps=ResultCaps(has_1d=True, has_2d=False),
            geom=GeomState(calibrated=True, energy_known=True),
        )
    )

    assert fields[FieldId.PROJECT_ROOT].status == StatusKind.OK
    assert fields[FieldId.PROJECT_ROOT].value == "/data"
    assert fields[FieldId.SOURCE_PATH].status == StatusKind.OK
    assert fields[FieldId.SOURCE_FRAMES].value == "3"
    assert fields[FieldId.CALIBRATION_PONI].status == StatusKind.OK
    assert fields[FieldId.OUTPUT_SAVE_PATH].value == "/data/out"
    assert fields[FieldId.RESULT_1D].status == StatusKind.OK
    assert fields[FieldId.RESULT_2D].status == StatusKind.MISSING


def test_energy_conflict_blocks_runs_from_field_status():
    state = ControlState(
        tool=Tool.INT_2D,
        source_caps=SourceCaps(has_frames=True, has_energy=True),
        geom=GeomState(
            calibrated=True,
            energy_known=True,
            calibration_energy_eV=12_700.0,
            source_energy_eV=13_100.0,
        ),
    )
    fields = build_field_statuses(state)
    profile = build_control_profile(state)

    assert fields[FieldId.BEAM_ENERGY].status == StatusKind.CONFLICT
    assert "disagrees" in fields[FieldId.BEAM_ENERGY].reason
    assert not profile.run_enabled
    assert profile.run_blockers == (fields[FieldId.BEAM_ENERGY].reason,)


def test_run_blockers_are_derived_from_required_field_statuses():
    state = ControlState(tool=Tool.RSM, mode=MeasMode.GI)
    fields = build_field_statuses(state)

    assert run_required_fields_for(state) == (
        FieldId.SOURCE_FRAMES,
        FieldId.CALIBRATION_PONI,
        FieldId.BEAM_ENERGY,
        FieldId.SAMPLE_ORIENTATION,
        FieldId.SOURCE_MOTORS,
        FieldId.DIFFRACTOMETER_UB,
    )
    assert run_blockers_from_fields(state, fields) == (
        "Choose a frame source.",
        "Load detector calibration.",
        "Set calibration wavelength or source energy.",
        "Set GI sample orientation.",
        "RSM needs motor metadata.",
        "Set/refine UB matrix before RSM.",
        "RSM GUI awaits real-data gate.",
    )


def test_control_profile_blockers_match_field_status_contract():
    states = [
        (
            ControlState(
                tool=Tool.INT_2D,
                source_caps=SourceCaps(has_frames=True, has_energy=True),
                geom=GeomState(
                    calibrated=True,
                    energy_known=True,
                    calibration_energy_eV=12_700.0,
                    source_energy_eV=13_100.0,
                ),
            ),
            FieldId.BEAM_ENERGY,
        ),
        (
            ControlState(
                tool=Tool.STITCH,
                mode=MeasMode.GI,
                backend="multigeometry",
                real_data_gates=frozenset({"gi_stitch_real_data"}),
                source_caps=SourceCaps(has_frames=True, has_energy=True),
                geom=GeomState(
                    calibrated=True,
                    energy_known=True,
                    gi_enabled=True,
                    sample_orientation_known=True,
                ),
            ),
            FieldId.PROCESSING_BACKEND,
        ),
        (ControlState(tool=Tool.INT_2D), None),
    ]

    for state, conflict_field in states:
        fields = build_field_statuses(state)
        profile = build_control_profile(state)

        assert profile.run_blockers == run_blockers_from_fields(state, fields)
        if conflict_field is not None:
            conflict = fields[conflict_field]
            assert conflict.status == StatusKind.CONFLICT
            assert conflict.reason in profile.run_blockers


def test_control_profile_returns_fields_by_section_in_inventory_order():
    profile = build_control_profile(
        ControlState(
            source_caps=SourceCaps(has_frames=True),
            processing_mode="Int 1D",
        )
    )

    source_fields = profile.fields_for(SectionId.SOURCE)
    project_fields = profile.fields_for(SectionId.PROJECT)
    assert [field.field_id for field in project_fields] == [FieldId.PROJECT_ROOT]
    assert [field.field_id for field in source_fields][:2] == [
        FieldId.SOURCE_PATH,
        FieldId.SOURCE_FRAMES,
    ]
    assert all(field.section == SectionId.SOURCE for field in source_fields)


def test_bound_control_state_describes_image_directory_form():
    state = build_bound_control_state(
        {
            ("Project", "project_folder"): "/data",
            ("Project", "h5_dir"): "/data/out",
            ("Signal", "poni_file"): "/data/cal.poni",
            ("Signal", "inp_type"): "Image Directory",
            ("Signal", "img_dir"): "/data/images",
            ("Signal", "img_ext"): "tif",
            ("Signal", "include_subdir"): True,
            ("Signal", "Filter"): "scan_",
            ("Signal", "meta_ext"): "SPEC",
            ("Signal", "meta_dir"): "/data/spec",
            ("Signal", "mask_file"): "/data/mask.edf",
            ("GI", "Grazing"): True,
            ("GI", "th_motor"): "Manual",
            ("GI", "th_val"): 0.1,
            ("GI", "sample_orientation"): 4,
            ("GI", "tilt_angle"): 0.0,
            ("MaskSat", "mask_sentinel"): False,
            ("BG", "bg_type"): "File",
            ("BG", "File"): "/data/bg.tif",
            ("BG", "Scale"): 1.0,
        },
        {
            ("Signal", "inp_type"): ("Image Series", "Image Directory"),
            ("GI", "th_motor"): ("th", "Manual"),
        },
    )

    source = state.fields_for(SectionId.SOURCE)
    experiment = state.fields_for(SectionId.EXPERIMENT)
    processing = state.fields_for(SectionId.PROCESSING)

    assert [field.label for field in source] == [
        "Source",
        "Directory",
        "File Type",
        "Subdirs",
        "Filter",
        "Meta Type",
        "SPEC Dir",
    ]
    assert source[0].kind == ControlFieldKind.COMBO
    assert source[0].choices == ("Image Series", "Image Directory")
    assert "Poni" in [field.label for field in experiment]
    assert "Mask File" in [field.label for field in experiment]
    assert any(field.label == "Theta" for field in experiment)
    assert [field.label for field in processing] == [
        "Mask Saturated",
        "Background",
        "BG File",
        "Scale",
    ]


def test_average_scan_is_a_processing_conditioning_field_not_source():
    """Average Scan moved from SOURCE to PROCESSING (design doc item 6).

    It is a frame-averaging *processing* choice, so it renders as a Conditioning
    bool next to Mask Saturated, not as a source-identity field.  Shown for any
    multi-frame image source (Image Series OR Image Directory), hidden for Single
    Image — matching the legacy wrangler's set_inp_type.
    """
    series = build_bound_control_state(
        {
            ("Signal", "inp_type"): "Image Series",
            ("Signal", "File"): "/data/img_0001.tif",
            ("Signal", "series_average"): False,
            ("MaskSat", "mask_sentinel"): False,
        },
        {("Signal", "inp_type"): ("Image Series", "Image Directory", "Single Image")},
    )
    source_paths = [field.path for field in series.fields_for(SectionId.SOURCE)]
    processing = series.fields_for(SectionId.PROCESSING)
    assert ("Signal", "series_average") not in source_paths
    avg = next(
        (f for f in processing if f.path == ("Signal", "series_average")), None
    )
    assert avg is not None
    assert avg.label == "Average Scan"
    assert avg.kind == ControlFieldKind.BOOL
    # Ordering: Average Scan follows Mask Saturated so the two coalesce into one
    # Conditioning pill row in the panel.
    labels = [field.label for field in processing]
    assert labels.index("Average Scan") == labels.index("Mask Saturated") + 1

    # Image Directory is also a multi-frame source (averaging applies within a
    # series) -> Average Scan still shown in PROCESSING, not SOURCE.
    directory = build_bound_control_state({
        ("Signal", "inp_type"): "Image Directory",
        ("Signal", "img_dir"): "/data/images",
        ("Signal", "series_average"): False,
        ("MaskSat", "mask_sentinel"): False,
    })
    assert "Average Scan" in [
        field.label for field in directory.fields_for(SectionId.PROCESSING)
    ]
    assert "Average Scan" not in [
        field.label for field in directory.fields_for(SectionId.SOURCE)
    ]

    # Single Image has no series to average -> hidden in BOTH sections.
    single = build_bound_control_state(
        {
            ("Signal", "inp_type"): "Single Image",
            ("Signal", "File"): "/data/img.tif",
            ("Signal", "series_average"): False,
        },
    )
    assert "Average Scan" not in [
        field.label for field in single.fields_for(SectionId.PROCESSING)
    ]
    assert "Average Scan" not in [
        field.label for field in single.fields_for(SectionId.SOURCE)
    ]


def test_gi_mode_overrides_stale_hidden_radial_label():
    """P2 (GI label staleness): in GI polar modes the integrator HIDES the radial
    label without resetting its text, so the stale widget text (e.g. 'Qip' left
    over from q_ip) must NOT win over the authoritative gi_mode."""
    from xdart.gui.tabs.static_scan.controls_logic import (
        _range_axis_labels_1d,
        _range_axis_labels_2d,
    )

    # 1D q_total carrying a STALE 'Qip' radial label -> gi_mode wins -> polar Q.
    radial, azim = _range_axis_labels_1d({
        ("Int1D", "gi_mode"): "q_total",
        ("Int1D", "unit"): "q_A^-1",
        ("Int1D", "radial_label"): "Qip (Å⁻¹)",
        ("Int1D", "azim_label"): "χ (°)",
    })
    assert radial == "Q (Å⁻¹)"
    assert azim == "χ (°)"

    # 2D q_chi carrying a stale 'Qip' radial label -> gi_mode wins.
    radial2, azim2 = _range_axis_labels_2d({
        ("Int2D", "gi_mode"): "q_chi",
        ("Int2D", "unit"): "q_A^-1",
        ("Int2D", "radial_label"): "Qip (Å⁻¹)",
        ("Int2D", "azim_label"): "χ (°)",
    })
    assert radial2 == "Q (Å⁻¹)"
    assert azim2 == "χ (°)"

    # Standard mode (no gi_mode) STILL prefers the live legacy label.
    radial_std, azim_std = _range_axis_labels_1d({
        ("Int1D", "radial_label"): "2θ (°)",
        ("Int1D", "azim_label"): "χ (°)",
    })
    assert radial_std == "2θ (°)"
    assert azim_std == "χ (°)"


def test_controls_logic_imports_no_heavy_deps():
    """controls_logic.py MUST stay Qt-free (CLAUDE.md: pure decision layer).

    Import it by dotted path in a clean subprocess and assert it does not pull
    the Qt/pyFAI GUI stack through ``static_scan.__init__``.  Then load the file
    itself in isolation and assert the module has no broad heavy imports. Mirrors
    ``test_display_logic_imports_no_heavy_deps``; makes the Qt-free invariant a
    CI gate rather than a convention.
    """
    import subprocess
    import sys
    import textwrap

    qt_forbidden = ('PySide6', 'PySide2', 'PyQt5', 'PyQt6',
                    'pyqtgraph', 'pyFAI')
    dotted = textwrap.dedent(f"""
        import sys
        import xdart.gui.tabs.static_scan.controls_logic
        bad = [m for m in {qt_forbidden!r} if m in sys.modules]
        if bad:
            print(','.join(bad))
            sys.exit(1)
    """)
    proc = subprocess.run([sys.executable, '-c', dotted],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"controls_logic pulled in Qt/pyFAI modules: {proc.stdout.strip()}\n"
        f"{proc.stderr.strip()}"
    )

    import xdart.gui.tabs.static_scan.controls_logic as _cl

    file_forbidden = ('PySide6', 'PySide2', 'PyQt5', 'PyQt6',
                      'pyqtgraph', 'h5py', 'pyFAI', 'fabio')
    isolated = textwrap.dedent(f"""
        import sys, importlib.util
        spec = importlib.util.spec_from_file_location(
            "controls_logic_isolated", {_cl.__file__!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        bad = [m for m in {file_forbidden!r} if m in sys.modules]
        if bad:
            print(','.join(bad))
            sys.exit(1)
    """)
    proc = subprocess.run([sys.executable, '-c', isolated],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"controls_logic module pulled in forbidden modules: {proc.stdout.strip()}\n"
        f"{proc.stderr.strip()}"
    )


def test_bound_control_state_describes_integration_form():
    state = build_bound_control_state(
        {
            ("Int1D", "axis"): "Q (Å⁻¹)",
            ("Int1D", "points"): "2000",
            ("Int1D", "radial_auto"): True,
            ("Int1D", "radial_low"): "0",
            ("Int1D", "radial_high"): "5",
            ("Int1D", "azim_auto"): True,
            ("Int1D", "azim_low"): "-180",
            ("Int1D", "azim_high"): "180",
            ("Int2D", "axis"): "Q-χ",
            ("Int2D", "radial_points"): "500",
            ("Int2D", "azim_points"): "500",
            ("Int2D", "radial_auto"): True,
            ("Int2D", "radial_low"): "0",
            ("Int2D", "radial_high"): "5",
            ("Int2D", "azim_auto"): True,
            ("Int2D", "azim_low"): "-180",
            ("Int2D", "azim_high"): "180",
        },
        {
            ("Int1D", "axis"): ("Q (Å⁻¹)", "2θ (°)", "χ (°)"),
            ("Int2D", "axis"): ("Q-χ", "2θ-χ"),
        },
    )

    processing = state.fields_for(SectionId.PROCESSING)
    by_path = {field.path: field for field in processing}

    assert by_path[("Int1D", "axis")].kind == ControlFieldKind.COMBO
    assert by_path[("Int1D", "axis")].choices == (
        "Q (Å⁻¹)", "2θ (°)", "χ (°)")
    assert by_path[("Int1D", "radial_auto")].kind == ControlFieldKind.BOOL
    assert by_path[("Int1D", "radial_low")].label == "Q (Å⁻¹) Low"
    assert by_path[("Int1D", "azim_low")].label == "χ (°) Low"
    assert not by_path[("Int1D", "radial_low")].enabled
    assert not by_path[("Int1D", "azim_low")].enabled
    assert by_path[("Int2D", "radial_points")].label == "2D Radial Points"
    assert by_path[("Int2D", "radial_low")].label == "Q (Å⁻¹) Low"
    assert by_path[("Int2D", "azim_low")].label == "χ (°) Low"
    assert not by_path[("Int2D", "radial_low")].enabled
    assert not by_path[("Int2D", "azim_low")].enabled


def test_bound_control_state_uses_axis_labels_for_range_rows():
    values = {
        ("Int1D", "axis"): "2θ (°)",
        ("Int1D", "radial_low"): "0",
        ("Int1D", "azim_low"): "-180",
        ("Int2D", "axis"): "Qᵢₚ–Qₒₒₚ",
        ("Int2D", "radial_low"): "-10",
        ("Int2D", "azim_low"): "0",
    }

    state = build_bound_control_state(values, tool=Tool.INT_2D)
    by_path = {
        field.path: field
        for field in state.fields_for(SectionId.PROCESSING)
    }

    assert by_path[("Int1D", "radial_low")].label == "2θ (°) Low"
    assert by_path[("Int1D", "azim_low")].label == "χ (°) Low"
    assert by_path[("Int2D", "radial_low")].label == "Qip (Å⁻¹) Low"
    assert by_path[("Int2D", "azim_low")].label == "Qoop (Å⁻¹) Low"


def test_bound_control_state_uses_hidden_gi_modes_for_legacy_labels():
    values = {
        ("Int1D", "axis"): "Qₒₒₚ",
        ("Int1D", "unit"): "q_A^-1",
        ("Int1D", "gi_mode"): "q_oop",
        ("Int1D", "radial_low"): "-10",
        ("Int1D", "azim_low"): "0",
        ("Int2D", "axis"): "Exit",
        ("Int2D", "unit"): "q_A^-1",
        ("Int2D", "gi_mode"): "exit_angles",
        ("Int2D", "radial_low"): "-5",
        ("Int2D", "azim_low"): "0",
    }

    state = build_bound_control_state(values, tool=Tool.INT_2D)
    by_path = {
        field.path: field
        for field in state.fields_for(SectionId.PROCESSING)
    }

    assert by_path[("Int1D", "radial_low")].label == "Qip (Å⁻¹) Low"
    assert by_path[("Int1D", "azim_low")].label == "Qoop (Å⁻¹) Low"
    assert by_path[("Int2D", "radial_low")].label == "Qip (Å⁻¹) Low"
    assert by_path[("Int2D", "azim_low")].label == "Exit (°) Low"


def test_bound_control_state_disables_locked_and_dependent_fields():
    values = {
        ("Mask", "Threshold"): False,
        ("Mask", "min"): "0",
        ("Int1D", "radial_auto"): True,
        ("Int1D", "radial_low"): "0",
        ("Int1D", "radial_high"): "5",
    }

    state = build_bound_control_state(values, tool=Tool.INT_1D)
    by_path = {
        field.path: field
        for field in state.fields_for(SectionId.PROCESSING)
    }

    assert by_path[("Mask", "Threshold")].enabled
    assert not by_path[("Mask", "min")].enabled
    assert "Threshold" in by_path[("Mask", "min")].reason
    assert not by_path[("Int1D", "radial_low")].enabled
    assert "Auto" in by_path[("Int1D", "radial_low")].reason

    locked = build_bound_control_state(
        values,
        tool=Tool.INT_1D,
        controls_enabled=False,
    )
    assert not any(field.enabled for field in locked.fields)


def test_bound_control_state_gates_integration_rows_by_tool():
    values = {
        ("Int1D", "axis"): "Q",
        ("Int1D", "points"): "2000",
        ("Int1D", "radial_auto"): True,
        ("Int1D", "radial_low"): "0",
        ("Int1D", "radial_high"): "5",
        ("Int1D", "azim_auto"): True,
        ("Int1D", "azim_low"): "-180",
        ("Int1D", "azim_high"): "180",
        ("Int2D", "axis"): "Q-χ",
        ("Int2D", "radial_points"): "500",
        ("Int2D", "azim_points"): "500",
        ("Int2D", "radial_auto"): True,
        ("Int2D", "radial_low"): "0",
        ("Int2D", "radial_high"): "5",
        ("Int2D", "azim_auto"): True,
        ("Int2D", "azim_low"): "-180",
        ("Int2D", "azim_high"): "180",
    }

    int_1d = build_bound_control_state(values, tool=Tool.INT_1D)
    int_2d = build_bound_control_state(values, tool=Tool.INT_2D)

    labels_1d = {
        field.label for field in int_1d.fields_for(SectionId.PROCESSING)
    }
    labels_2d = {
        field.label for field in int_2d.fields_for(SectionId.PROCESSING)
    }

    assert "1D Axis" in labels_1d
    assert "1D Points" in labels_1d
    assert "2D Axis" not in labels_1d
    assert "1D Axis" in labels_2d
    assert "1D Points" in labels_2d
    assert "2D Axis" in labels_2d
    assert "2D Radial Points" in labels_2d


def test_bound_control_paths_cover_transitional_sections():
    assert ("Project", "project_folder") in BOUND_CONTROL_PATHS
    assert ("Signal", "inp_type") in BOUND_CONTROL_PATHS
    assert ("NeXus File", "nexus_file") in BOUND_CONTROL_PATHS
    assert ("GI", "Grazing") in BOUND_CONTROL_PATHS
    assert ("BG", "bg_type") in BOUND_CONTROL_PATHS
    assert ("Int1D", "axis") in INTEGRATION_CONTROL_PATHS
    assert ("Int2D", "azim_points") in INTEGRATION_CONTROL_PATHS


def test_integrator_binding_table_is_the_single_membership_source():
    bound_paths = set(BOUND_CONTROL_PATHS)
    backed_paths = {spec.path for spec in INTEGRATOR_BACKED_CONTROL_SPECS}

    assert set(INTEGRATOR_BACKED_CONTROL_PATHS) == backed_paths
    assert set(INTEGRATION_CONTROL_PATHS) == {
        path for path in backed_paths if path[0] in {"Int1D", "Int2D"}
    }
    assert backed_paths <= bound_paths


def test_coerce_control_edit_value_matches_backing_type():
    assert coerce_control_edit_value(True, "false") is False
    assert coerce_control_edit_value(False, "yes") is True
    assert coerce_control_edit_value(4, "5.0") == 5
    assert coerce_control_edit_value(1.5, "2.25") == 2.25
    assert coerce_control_edit_value("old", "new") == "new"


def test_bound_control_state_describes_nexus_form_without_qt():
    state = build_bound_control_state(
        {
            ("Project", "project_folder"): "/data",
            ("Output", "h5_dir"): "/data/out",
            ("Calibration", "poni_file"): "/data/cal.poni",
            ("NeXus File", "nexus_file"): "/data/scan.nxs",
            ("NeXus File", "entry"): "entry",
            ("GI", "Grazing"): False,
            ("GI", "th_motor"): "th",
            ("GI", "sample_orientation"): 4,
            ("GI", "tilt_angle"): 0.0,
            ("MaskSat", "mask_sentinel"): False,
            ("BG", "bg_type"): "None",
        }
    )

    assert [field.label for field in state.fields_for(SectionId.PROJECT)] == [
        "Folder",
        "Save Path",
    ]
    assert [field.label for field in state.fields_for(SectionId.SOURCE)] == [
        "NeXus File",
        "Entry",
    ]
    assert "Poni" in [
        field.label for field in state.fields_for(SectionId.EXPERIMENT)
    ]
    assert "Theta" not in [
        field.label for field in state.fields_for(SectionId.EXPERIMENT)
    ]


def test_control_panel_state_combines_profile_and_bound_fields():
    state = build_control_panel_state(
        ControlState(
            tool=Tool.INT_2D,
            source_caps=SourceCaps(has_frames=True),
            geom=GeomState(calibrated=True, energy_known=True),
        ),
        {
            ("Project", "project_folder"): "/data",
            ("Project", "h5_dir"): "/data/out",
            ("Signal", "inp_type"): "Single Image",
        },
    )

    assert state.profile.processing_page.value == "int_2d"
    assert state.profile.can_run
    assert state.bound_controls is not None
    assert state.bound_controls.value_for(
        ("Project", "project_folder")
    ) == "/data"


def test_control_profile_exposes_section_actions():
    profile = build_control_profile(
        ControlState(
            tool=Tool.INT_2D,
            source_caps=SourceCaps(has_frames=True),
            geom=GeomState(calibrated=True, energy_known=True),
        )
    )

    source_actions = profile.actions_for(SectionId.SOURCE)
    project_actions = profile.actions_for(SectionId.PROJECT)
    experiment_actions = profile.actions_for(SectionId.EXPERIMENT)
    processing_actions = profile.actions_for(SectionId.PROCESSING)

    assert [action.action for action in project_actions] == [
        ControlAction.CHOOSE_PROJECT,
        ControlAction.CHOOSE_OUTPUT,
    ]
    assert [action.action for action in source_actions] == [
        ControlAction.CHOOSE_SOURCE]
    # Refine is hidden in the plain 1-D/2-D integration modes (here INT_2D).
    assert [action.action for action in experiment_actions] == [
        ControlAction.CALIBRATE,
        ControlAction.MAKE_MASK,
    ]
    # ...but it returns for Stitch/RSM/GI, still disabled pending its gate.
    refine_actions = build_control_profile(
        ControlState(
            tool=Tool.STITCH,
            source_caps=SourceCaps(has_frames=True),
            geom=GeomState(calibrated=True, energy_known=True),
        )
    ).actions_for(SectionId.EXPERIMENT)
    assert [action.action for action in refine_actions] == [
        ControlAction.CALIBRATE,
        ControlAction.MAKE_MASK,
        ControlAction.REFINE_GEOMETRY,
    ]
    assert not refine_actions[-1].enabled
    assert [action.action for action in processing_actions] == [
        ControlAction.REINTEGRATE_1D,
        ControlAction.REINTEGRATE_2D,
        ControlAction.ADVANCED_PROCESSING,
    ]
    assert all(action.enabled for action in processing_actions)


def test_viewer_profile_disables_advanced_processing_action():
    profile = build_control_profile(ControlState(tool=Tool.IMAGE_VIEWER))
    action = profile.actions_for(SectionId.PROCESSING)[0]
    assert action.action == ControlAction.ADVANCED_PROCESSING
    assert not action.enabled
