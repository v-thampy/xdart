from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

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


def test_vnext_edit_commits_to_store_and_freezes_same_revision(tmp_path):
    original_root = tmp_path / "before"
    edited_root = tmp_path / "after"
    store = RunIntentStore(RunIntent(project_root=str(original_root)))

    captured = store.snapshot()
    before = project_controls(captured, None, RunPhase.IDLE)
    assert before.bound_controls is not None
    assert before.bound_controls.value_for(PROJECT_ROOT) == str(original_root)

    candidate = reduce_control_edit(captured, PROJECT_ROOT, str(edited_root))
    assert isinstance(candidate, RunIntent)
    assert store.snapshot().thaw().project_root == str(original_root)

    committed = store.commit(candidate, expected_revision=captured.revision)
    assert isinstance(committed, IntentCommitAccepted)
    projected = project_controls(committed.snapshot, None, RunPhase.IDLE)
    assert projected.bound_controls is not None
    assert projected.bound_controls.value_for(PROJECT_ROOT) == str(edited_root)

    frozen = store.freeze(expected_revision=committed.revision)
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.project_root == str(edited_root)
    assert frozen.configuration.save_path == str(
        edited_root / "xdart_processed_data"
    )
