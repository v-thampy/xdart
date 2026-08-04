from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.controls_projection import (
    EditRefusal,
    GI_ENABLED,
    GI_MOTOR,
    GI_ORIENTATION,
    GI_THETA,
    GI_TILT,
    project_controls,
    reduce_control_edit,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
    ControlsPanelV2,
    FormRow,
    SubsectionCard,
)
from xrd_tools.session.readiness import (
    ControlAction,
    SectionId,
    build_native_int_reduction_plan_from_args,
)
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/raw/eiger"),
            recursive=True,
            suffixes=(".h5",),
            name_filter="scan",
        ),
        project_root="/raw/eiger",
        save_path="/processed",
        poni_file="/calibration/detector.poni",
        mask_file="/calibration/mask.edf",
        output_mode="Overwrite",
        bai_1d_args={
            "npt": 128,
            "method": "csr",
            "radial_range": (0.2, 4.2),
        },
        bai_2d_args={
            "npt_rad": 64,
            "npt_azim": 32,
            "method": "csr",
            "azimuth_range": (-90.0, 90.0),
        },
    )


def test_production_projection_supplies_source_and_integration_inventory() -> None:
    state = project_controls(
        RunIntentStore(_intent()).snapshot(),
        None,
        RunPhase.IDLE,
    )
    bound = state.bound_controls
    assert bound is not None
    fields = {field.path: field for field in bound.fields}

    required = {
        ("Signal", "inp_type"),
        ("Signal", "img_dir"),
        ("Signal", "include_subdir"),
        ("Signal", "img_ext"),
        ("Signal", "Filter"),
        ("Int1D", "axis"),
        ("Int1D", "points"),
        ("Int1D", "radial_auto"),
        ("Int1D", "radial_low"),
        ("Int1D", "radial_high"),
        ("Int1D", "azim_auto"),
        ("Int1D", "azim_low"),
        ("Int1D", "azim_high"),
        ("Int2D", "axis"),
        ("Int2D", "radial_points"),
        ("Int2D", "azim_points"),
        ("Int2D", "radial_auto"),
        ("Int2D", "radial_low"),
        ("Int2D", "radial_high"),
        ("Int2D", "azim_auto"),
        ("Int2D", "azim_low"),
        ("Int2D", "azim_high"),
        ("Mask", "Threshold"),
        ("MaskSat", "mask_sentinel"),
        ("Signal", "series_average"),
        ("BG", "bg_type"),
    }
    assert required <= fields.keys()
    assert fields[("Signal", "inp_type")].value == "Image Directory"
    assert fields[("Signal", "img_dir")].value == "/raw/eiger"
    assert fields[("Signal", "include_subdir")].value is True
    assert fields[("Signal", "img_ext")].value == "h5"
    assert fields[("Signal", "Filter")].value == "scan"
    assert ("Source", "energy_preference") not in fields
    assert fields[("Int1D", "axis")].value == "Q (Å⁻¹)"
    assert fields[("Int1D", "points")].value == 128
    assert fields[("Int1D", "radial_auto")].value is False
    assert fields[("Int1D", "radial_low")].value == 0.2
    assert fields[("Int1D", "radial_high")].value == 4.2
    assert fields[("Int1D", "azim_auto")].value is True
    assert fields[("Int2D", "radial_points")].value == 64
    assert fields[("Int2D", "azim_points")].value == 32
    assert fields[("Int2D", "radial_auto")].value is True
    assert fields[("Int2D", "azim_auto")].value is False
    assert fields[("Int2D", "azim_low")].value == -90.0
    assert fields[("Int2D", "azim_high")].value == 90.0
    assert fields[("Signal", "series_average")].enabled is False
    assert fields[("BG", "bg_type")].enabled is False
    actions = state.profile.actions_for(SectionId.PROCESSING)
    assert tuple(action.action for action in actions) == (
        ControlAction.REINTEGRATE_1D,
        ControlAction.REINTEGRATE_2D,
        ControlAction.ADVANCED_PROCESSING,
    )
    assert all(not action.enabled and action.reason for action in actions)
    experiment_actions = state.profile.actions_for(SectionId.EXPERIMENT)
    assert tuple(action.action for action in experiment_actions) == (
        ControlAction.CALIBRATE,
        ControlAction.MAKE_MASK,
    )
    assert all(
        not action.enabled and action.reason
        for action in experiment_actions
    )


def test_gi_detail_fields_are_inline_only_in_grazing_mode() -> None:
    gi_paths = {GI_MOTOR, GI_THETA, GI_ORIENTATION, GI_TILT}
    intent = _intent()
    standard = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    assert standard.bound_controls is not None
    assert not gi_paths & {
        candidate.path for candidate in standard.bound_controls.fields
    }

    intent.gi.enabled = True
    grazing = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    assert grazing.bound_controls is not None
    grazing_fields = {
        candidate.path: candidate
        for candidate in grazing.bound_controls.fields
    }
    assert gi_paths <= grazing_fields.keys()
    assert grazing_fields[GI_MOTOR].value == "Manual"
    assert grazing_fields[GI_THETA].value == 0.1

    intent.gi.incidence_motor = "th"
    named = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    assert named.bound_controls is not None
    named_paths = {
        candidate.path for candidate in named.bound_controls.fields
    }
    assert GI_MOTOR in named_paths
    assert GI_THETA not in named_paths
    assert {GI_ORIENTATION, GI_TILT} <= named_paths


def test_gi_main_surface_has_one_points_owner_and_edit_resynchronizes_grid(
) -> None:
    intent = _intent()
    intent.gi.enabled = True
    intent.bai_1d_args["npt_oop"] = 37
    snapshot = RunIntentStore(intent).snapshot()

    state = project_controls(snapshot, None, RunPhase.IDLE)
    assert state.bound_controls is not None
    fields = {
        candidate.path: candidate
        for candidate in state.bound_controls.fields
    }
    assert fields[("Int1D", "points")].value == 128
    assert ("Int1D", "points_oop") not in fields
    # Projection is passive: an asymmetric value loaded from an existing
    # session survives for a future Advanced editor.
    assert snapshot.thaw().bai_1d_args["npt_oop"] == 37

    resynchronized = reduce_control_edit(
        snapshot, ("Int1D", "points"), "128"
    )
    assert type(resynchronized) is RunIntent
    assert resynchronized.bai_1d_args["npt"] == 128
    assert resynchronized.bai_1d_args["npt_oop"] == 128

    changed = reduce_control_edit(snapshot, ("Int1D", "points"), "256")
    assert type(changed) is RunIntent
    assert changed.bai_1d_args["npt"] == 256
    assert changed.bai_1d_args["npt_oop"] == 256
    plan = build_native_int_reduction_plan_from_args(
        changed.bai_1d_args,
        changed.bai_2d_args,
        gi_enabled=True,
        gi_incident_angle=0.1,
    )
    assert plan.integration_1d is not None
    assert plan.integration_1d.npt == 256
    assert plan.gi is not None
    assert plan.gi.npt_oop == 256
    assert plan.gi.method == "cython"


def test_standard_grazing_schema_change_rebuilds_exact_axis_choices() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    intent = _intent()
    standard = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    intent.gi.enabled = True
    grazing = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    panel = ControlsPanelV2()
    try:
        panel.set_state(standard)
        standard_axis = next(
            row
            for row in panel.findChildren(FormRow)
            if row.path == ("Int1D", "axis")
        )
        assert tuple(
            standard_axis.editor.itemText(index)
            for index in range(standard_axis.editor.count())
        ) == ("Q (Å⁻¹)", "2θ (°)", "χ (°)")

        # Grazing adds fields, so the in-place path must reject the stale
        # schema and let the shell reconstruct it.
        assert panel.apply_state_update(grazing) is False
        panel.set_state(grazing)
        app.processEvents()
        grazing_axis = next(
            row
            for row in panel.findChildren(FormRow)
            if row.path == ("Int1D", "axis")
        )
        assert grazing_axis is not standard_axis
        assert tuple(
            grazing_axis.editor.itemText(index)
            for index in range(grazing_axis.editor.count())
        ) == ("Q", "Qip", "Qoop", "Exit", "χGI")
        assert not any(
            row.path == ("Int1D", "points_oop")
            for row in panel.findChildren(FormRow)
        )

        # Standard removes the GI detail fields and must likewise reconstruct,
        # with no stale GI axis choices surviving the transition.
        assert panel.apply_state_update(standard) is False
        panel.set_state(standard)
        app.processEvents()
        restored_axis = next(
            row
            for row in panel.findChildren(FormRow)
            if row.path == ("Int1D", "axis")
        )
        assert tuple(
            restored_axis.editor.itemText(index)
            for index in range(restored_axis.editor.count())
        ) == ("Q (Å⁻¹)", "2θ (°)", "χ (°)")
    finally:
        panel.close()


def test_integration_edits_replace_only_the_named_next_run_value() -> None:
    store = RunIntentStore(_intent())
    snapshot = store.snapshot()

    points = reduce_control_edit(
        snapshot,
        ("Int1D", "points"),
        "256",
    )
    assert type(points) is RunIntent
    assert points.bai_1d_args["npt"] == 256
    assert points.bai_1d_args["method"] == "csr"
    assert snapshot.thaw().bai_1d_args["npt"] == 128

    axis = reduce_control_edit(
        snapshot,
        ("Int1D", "axis"),
        "2θ (°)",
    )
    assert type(axis) is RunIntent
    assert axis.bai_1d_args["unit"] == "2th_deg"

    auto = reduce_control_edit(
        snapshot,
        ("Int2D", "radial_auto"),
        False,
    )
    assert type(auto) is RunIntent
    assert auto.bai_2d_args["radial_range"] == (0.0, 5.0)

    invalid = reduce_control_edit(
        snapshot,
        ("Int2D", "azim_points"),
        0,
    )
    assert isinstance(invalid, EditRefusal)


def test_integration_edit_freezes_into_exact_native_plan() -> None:
    snapshot = RunIntentStore(_intent()).snapshot()
    candidate = reduce_control_edit(snapshot, ("Int1D", "points"), 256)
    assert type(candidate) is RunIntent
    candidate = reduce_control_edit(
        RunIntentStore(candidate).snapshot(),
        ("Int2D", "radial_auto"),
        False,
    )
    assert type(candidate) is RunIntent
    frozen = candidate.freeze()
    plan = build_native_int_reduction_plan_from_args(
        frozen.bai_1d_args,
        frozen.bai_2d_args,
    )
    assert plan.integration_1d is not None
    assert plan.integration_1d.npt == 256
    assert plan.integration_1d.method == "csr"
    assert plan.integration_2d is not None
    assert plan.integration_2d.npt_rad == 64
    assert plan.integration_2d.npt_azim == 32
    assert plan.integration_2d.radial_range == (0.0, 5.0)


def test_axis_change_clears_only_the_incompatible_radial_range() -> None:
    snapshot = RunIntentStore(_intent()).snapshot()
    changed = reduce_control_edit(snapshot, ("Int1D", "axis"), "2θ (°)")
    assert type(changed) is RunIntent
    assert changed.bai_1d_args == {
        "npt": 128,
        "method": "csr",
        "unit": "2th_deg",
    }
    assert changed.bai_2d_args == snapshot.thaw().bai_2d_args


def test_gi_transition_normalizes_units_and_only_incompatible_ranges() -> None:
    intent = _intent()
    intent.bai_1d_args["unit"] = "2th_deg"
    intent.bai_2d_args.update({
        "unit": "q_A^-1",
        "radial_range": (0.1, 4.0),
    })
    enabled = reduce_control_edit(
        RunIntentStore(intent).snapshot(),
        GI_ENABLED,
        True,
    )
    assert type(enabled) is RunIntent
    assert enabled.gi.enabled is True
    assert enabled.bai_1d_args["unit"] == "q_A^-1"
    assert "radial_range" not in enabled.bai_1d_args
    assert enabled.bai_2d_args["unit"] == "qip_A^-1"
    assert "radial_range" not in enabled.bai_2d_args

    enabled.bai_1d_args["radial_range"] = (0.1, 3.0)
    disabled = reduce_control_edit(
        RunIntentStore(enabled).snapshot(),
        GI_ENABLED,
        False,
    )
    assert type(disabled) is RunIntent
    assert disabled.gi.enabled is False
    assert disabled.bai_1d_args["unit"] == "q_A^-1"
    assert disabled.bai_2d_args["unit"] == "q_A^-1"
    assert disabled.bai_1d_args["radial_range"] == (0.1, 3.0)
    assert "radial_range" not in disabled.bai_2d_args

    semantic = _intent()
    semantic.gi.mode_1d = "exit_angle"
    semantic.gi.mode_2d = "exit_angles"
    semantic.bai_1d_args["unit"] = "q_A^-1"
    semantic.bai_2d_args.update({
        "unit": "q_A^-1",
        "radial_range": (0.1, 4.0),
    })
    changed_semantic = reduce_control_edit(
        RunIntentStore(semantic).snapshot(),
        GI_ENABLED,
        True,
    )
    assert type(changed_semantic) is RunIntent
    assert "radial_range" not in changed_semantic.bai_1d_args
    assert "radial_range" not in changed_semantic.bai_2d_args


def test_nonzero_threshold_bound_enables_next_run_threshold() -> None:
    snapshot = RunIntentStore(_intent()).snapshot()
    changed = reduce_control_edit(snapshot, ("Mask", "max"), "1000")
    assert type(changed) is RunIntent
    assert changed.threshold.threshold_max == 1000.0
    assert changed.threshold.apply_threshold is True
    assert snapshot.thaw().threshold.apply_threshold is False

    zero = reduce_control_edit(snapshot, ("Mask", "max"), "0")
    assert type(zero) is RunIntent
    assert zero.threshold.threshold_max == 0.0
    assert zero.threshold.apply_threshold is False


def test_unrepresentable_suffix_tuple_is_exact_and_not_editable() -> None:
    intent = _intent()
    intent.source_spec = DirectorySourceSpec(
        Path("/raw/eiger"),
        suffixes=("_master.hdf5", "_master.h5"),
    )
    snapshot = RunIntentStore(intent).snapshot()
    state = project_controls(snapshot, None, RunPhase.IDLE)
    assert state.bound_controls is not None
    fields = {field.path: field for field in state.bound_controls.fields}
    suffix = fields[("Signal", "img_ext")]
    assert suffix.value == "_master.hdf5, _master.h5"
    assert suffix.enabled is False
    assert "complete source" in suffix.reason
    refused = reduce_control_edit(
        snapshot,
        ("Signal", "img_ext"),
        "h5",
    )
    assert isinstance(refused, EditRefusal)
    assert snapshot.thaw().source_spec == intent.source_spec


def test_invalid_integration_edits_leave_snapshot_exact() -> None:
    snapshot = RunIntentStore(_intent()).snapshot()
    before = snapshot.thaw()
    for path, value in (
        (("Int1D", "points"), -1),
        (("Int2D", "radial_points"), 0),
        (("Int1D", "radial_low"), float("nan")),
        (("Int2D", "azim_high"), float("inf")),
        (("Int1D", "radial_low"), 9.0),
    ):
        assert isinstance(reduce_control_edit(snapshot, path, value), EditRefusal)
        assert snapshot.thaw() == before


def test_source_edit_replaces_one_complete_immutable_source_without_io(
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("projection/edit must not enumerate or open source data")

    monkeypatch.setattr(Path, "iterdir", forbidden)
    snapshot = RunIntentStore(_intent()).snapshot()
    state = project_controls(snapshot, None, RunPhase.IDLE)
    assert state.bound_controls is not None
    changed = reduce_control_edit(
        snapshot,
        ("Signal", "img_dir"),
        "/raw/replacement",
    )
    assert type(changed) is RunIntent
    assert changed.source_spec == DirectorySourceSpec(
        Path("/raw/replacement"),
        recursive=True,
        suffixes=(".h5",),
        name_filter="scan",
        generation=1,
    )
    assert snapshot.thaw().source_spec == _intent().source_spec


def test_empty_integration_args_project_native_builder_defaults() -> None:
    intent = _intent()
    intent.bai_1d_args.clear()
    intent.bai_2d_args.clear()
    state = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    assert state.bound_controls is not None
    fields = {field.path: field.value for field in state.bound_controls.fields}
    assert fields[("Int1D", "points")] == 1000
    assert fields[("Int2D", "radial_points")] == 1000
    assert fields[("Int2D", "azim_points")] == 360


def test_production_projection_mounts_full_processing_subsections() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = ControlsPanelV2()
    panel.resize(520, 1000)
    panel.show()
    try:
        panel.set_state(
            project_controls(
                RunIntentStore(_intent()).snapshot(),
                None,
                RunPhase.IDLE,
            )
        )
        app.processEvents()
        paths = {
            tuple(row.path)
            for row in panel.findChildren(FormRow)
        }
        assert {
            ("Signal", "img_dir"),
            ("Int1D", "axis"),
            ("Int2D", "axis"),
            ("BG", "bg_type"),
        } <= paths
        titles = {
            card.title.text()
            for card in panel.findChildren(SubsectionCard)
            if card.title.isVisible()
        }
        assert {"1-D", "2-D", "Conditioning", "Background"} <= titles
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# LV-UI-5b — metadata-preferred incidence motor (F3 sticky Manual)
# ---------------------------------------------------------------------------

def _motor_page():
    from pyqtgraph.Qt import QtWidgets
    from xrd_tools.session.intent_store import RunIntentStore
    from xdart.gui.tabs.scattering.adapters.run_executor import (
        StandardRunExecutor,
    )
    from xdart.gui.tabs.scattering.adapters.source import (
        FilesystemSourceAdapter,
    )
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(output_mode="Overwrite")),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(),
    )


def _motor_knowledge(*choices):
    from types import SimpleNamespace

    # The seam consumes exactly ``gi_motor_choices``; the qualification of a
    # full SourceObservation is proven by the _on_observation rows above.
    return SimpleNamespace(gi_motor_choices=tuple(choices))


def test_metadata_motor_default_pick_adopts_the_preferred_motor():
    page = _motor_page()
    try:
        page._maybe_default_gi_motor(_motor_knowledge("exposure", "chi", "th"))
        assert page._intents.snapshot().thaw().gi.incidence_motor == "th"
    finally:
        page.close_workspace()
        page.deleteLater()


def test_metadata_motor_default_pick_respects_a_deliberate_manual():
    from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR

    page = _motor_page()
    try:
        page._maybe_default_gi_motor(_motor_knowledge("exposure", "chi", "th"))
        assert page._intents.snapshot().thaw().gi.incidence_motor == "th"
        # The user deliberately chooses Manual for this source...
        page._on_field_value(GI_MOTOR, "Manual")
        assert page._intents.snapshot().thaw().gi.incidence_motor == "Manual"
        # ...and a re-observation must NOT flip it back (F3 sticky rule).
        page._maybe_default_gi_motor(_motor_knowledge("exposure", "chi", "th"))
        assert page._intents.snapshot().thaw().gi.incidence_motor == "Manual"
    finally:
        page.close_workspace()
        page.deleteLater()
