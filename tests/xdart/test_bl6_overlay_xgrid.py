"""Guards for the retained production Waterfall row selector.

The old payload-owned accumulator never reached the current Scattering
Workspace.  Keep its retired API absent while pinning the one display helper
that ScientificView still imports and calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.session.display_logic import waterfall_display_rows


pytestmark = pytest.mark.display_logic

_ROOT = Path(__file__).resolve().parents[2]
_DISPLAY_LOGIC = _ROOT / "src/xrd_tools/session/display_logic.py"
_SCIENTIFIC_VIEW = _ROOT / "src/xdart/gui/tabs/scattering/scientific_view.py"
_DISPLAY_LOGIC_MODULE = "xrd_tools.session.display_logic"

_RETIRED = frozenset({
    "GridMismatchPolicy",
    "IncompatibleGridError",
    "RowMeta",
    "WaterfallHistory",
    "qualified_frame_id",
    "frame_index_from_qualified_id",
    "scan_key_from_qualified_id",
    "overlay_grid_reset_key",
    "overlay_grid_keys_compatible",
    "overlay_axes_compatible",
    "overlay_grid_summary",
    "accumulate_waterfall",
    "LifecycleCause",
    "LifecycleReset",
    "AccumulatorLifecycle",
    "accumulator_clearable",
    "normalize_rows",
    "display_grid_for_history",
    "render_waterfall_view",
    "_reset_keys_compatible",
    "_dedup_key",
    "_freeze_metadata",
    "_dedup_first",
    "_waterfall_capacity",
    "_new_waterfall_row_buffer",
    "_history_row_buffer",
    "_waterfall_history_from_buffer",
    "_fresh_waterfall_history",
    "_interp_overlap",
    "_radial_kind",
    "_display_grid_for_rows",
})


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_definitions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _literal_all(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, (ast.List, ast.Tuple))
        values = tuple(
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        )
        assert len(values) == len(node.value.elts)
        return values
    raise AssertionError("display_logic must retain one literal __all__")


def test_retired_accumulator_surface_and_source_callers_stay_absent() -> None:
    display_tree = _tree(_DISPLAY_LOGIC)
    definitions = _top_level_definitions(display_tree)
    exports = set(_literal_all(display_tree))

    assert "waterfall_display_rows" in definitions
    assert "waterfall_display_rows" in exports
    assert not (_RETIRED & definitions)
    assert not (_RETIRED & exports)

    violations = []
    for path in sorted((_ROOT / "src").rglob("*.py")):
        tree = _tree(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == _DISPLAY_LOGIC_MODULE
            ):
                for alias in node.names:
                    if alias.name == "*" or alias.name in _RETIRED:
                        violations.append((path.relative_to(_ROOT), alias.name))
            elif isinstance(node, ast.Attribute) and node.attr in _RETIRED:
                violations.append((path.relative_to(_ROOT), node.attr))
    assert violations == []


def test_scientific_view_keeps_one_waterfall_row_selector_call() -> None:
    tree = _tree(_SCIENTIFIC_VIEW)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == _DISPLAY_LOGIC_MODULE
        for alias in node.names
        if alias.name == "waterfall_display_rows"
    ]
    owners = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_bounded_waterfall_rows"
    ]

    assert imports == ["waterfall_display_rows"]
    assert len(owners) == 1
    calls = [
        node
        for node in ast.walk(owners[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "waterfall_display_rows"
    ]
    assert len(calls) == 1


def test_waterfall_display_rows_bounds_and_keeps_terminal_identity() -> None:
    rows = np.arange(651 * 3, dtype=float).reshape(651, 3)
    ids = tuple(("scan", index) for index in range(651))

    displayed, displayed_ids, indices = waterfall_display_rows(rows, ids, 256)

    assert indices is not None
    assert displayed.shape == (256, 3)
    assert len(displayed_ids) == len(indices) == 256
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 650
    assert np.all(np.diff(indices) > 0)
    np.testing.assert_array_equal(displayed, rows[indices])
    assert displayed_ids == tuple(ids[int(index)] for index in indices)

    terminal, terminal_ids, terminal_indices = waterfall_display_rows(
        rows, ids, 1
    )
    np.testing.assert_array_equal(terminal, rows[[-1]])
    assert terminal_ids == (ids[-1],)
    np.testing.assert_array_equal(terminal_indices, np.array([650]))
