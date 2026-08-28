from __future__ import annotations

import ast
from dataclasses import MISSING, fields
from pathlib import Path

import pytest

from xdart.gui.tabs.scattering import (
    controls_editing,
    controls_inventory,
    controls_projection,
)
from xdart.gui.tabs.scattering.controls_editing import reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import (
    CONTROL_FIELD_SPECS,
    PROJECT_ROOT,
    ControlFieldSpec,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentFreezeAccepted,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session import readiness
from xrd_tools.session.readiness import ControlsProjection


def test_native_control_schema_has_no_static_widget_metadata():
    assert {field.name for field in fields(ControlFieldSpec)} == {
        "section",
        "label",
        "path",
        "kind",
        "tools",
    }
    paths = tuple(spec.path for spec in CONTROL_FIELD_SPECS)
    assert len(paths) == len(set(paths))
    assert ("Int1D", "unit") not in paths
    assert ("Int2D", "unit") not in paths
    assert ("Int1D", "method") not in paths
    assert ("Int2D", "method") not in paths


def test_vnext_controls_modules_do_not_import_static_binding_schema():
    forbidden = {
        "StaticWidgetBinding",
        "INTEGRATION_CONTROL_SPECS",
        "INTEGRATOR_BACKED_CONTROL_PATHS",
        "INTEGRATOR_BACKED_CONTROL_SPECS",
    }
    for module in (
        controls_inventory,
        controls_projection,
        controls_editing,
    ):
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported.isdisjoint(forbidden), (
            f"{path.name} imports the static-page binding schema: "
            f"{sorted(imported & forbidden)}"
        )

    controls_renderer = (
        Path(controls_inventory.__file__).parents[2]
        / "widgets"
        / "controls_panel.py"
    )
    assert controls_renderer.is_file()
    renderer_text = controls_renderer.read_text(encoding="utf-8")
    assert "static_controls_adapter" not in renderer_text
    assert "StaticWidgetBinding" not in renderer_text


def test_controls_projection_has_one_required_native_owner_and_renderer_route():
    projection_fields = fields(ControlsProjection)
    assert tuple(field.name for field in projection_fields) == (
        "processing_page",
        "fields",
        "section_actions",
        "detector_summary",
    )
    assert all(field.default is MISSING for field in projection_fields)
    assert all(field.default_factory is MISSING for field in projection_fields)

    readiness_path = Path(readiness.__file__)
    readiness_tree = ast.parse(
        readiness_path.read_text(encoding="utf-8"),
        filename=str(readiness_path),
    )
    readiness_classes = {
        node.name
        for node in readiness_tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert readiness_classes.isdisjoint({
        "BoundControlState",
        "ControlPanelRenderState",
        "ControlProfile",
        "ControlState",
    })

    projection_path = Path(controls_projection.__file__)
    projection_tree = ast.parse(
        projection_path.read_text(encoding="utf-8"),
        filename=str(projection_path),
    )
    projected_names = {
        node.id for node in ast.walk(projection_tree)
        if isinstance(node, ast.Name)
    }
    assert "project_control_fields" in projected_names
    assert "build_native_control_state" not in projected_names

    controls_renderer = (
        Path(controls_inventory.__file__).parents[2]
        / "widgets"
        / "controls_panel.py"
    )
    renderer_tree = ast.parse(
        controls_renderer.read_text(encoding="utf-8"),
        filename=str(controls_renderer),
    )
    panel_class = next(
        node for node in renderer_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControlsPanel"
    )
    methods = {
        node.name for node in panel_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "reconcile" in methods
    assert methods.isdisjoint({
        "apply_state_update",
        "current_form_edits",
        "set_bound_state",
        "set_profile",
        "set_state",
    })

    run_controls_path = controls_renderer.with_name("run_controls.py")
    run_controls_text = run_controls_path.read_text(encoding="utf-8")
    assert "statusLabel" not in run_controls_text
    assert "def set_run_active" not in run_controls_text

    projected = project_controls(
        RunIntentStore(RunIntent()).snapshot(),
        None,
        RunPhase.IDLE,
    )
    assert type(projected) is ControlsProjection
    with pytest.raises(TypeError):
        projected.section_actions[next(iter(projected.section_actions))] = ()


def test_vnext_edit_commits_to_store_and_freezes_same_revision(tmp_path):
    original_root = tmp_path / "before"
    edited_root = tmp_path / "after"
    store = RunIntentStore(RunIntent(project_root=str(original_root)))

    captured = store.snapshot()
    before = project_controls(captured, None, RunPhase.IDLE)
    assert before.value_for(PROJECT_ROOT) == str(original_root)

    candidate = reduce_control_edit(captured, PROJECT_ROOT, str(edited_root))
    assert isinstance(candidate, RunIntent)
    assert store.snapshot().thaw().project_root == str(original_root)

    committed = store.commit(candidate, expected_revision=captured.revision)
    assert isinstance(committed, IntentCommitAccepted)
    projected = project_controls(committed.snapshot, None, RunPhase.IDLE)
    assert projected.value_for(PROJECT_ROOT) == str(edited_root)

    frozen = store.freeze(expected_revision=committed.revision)
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.project_root == str(edited_root)
    assert frozen.configuration.save_path == str(
        edited_root / "xdart_processed_data"
    )
