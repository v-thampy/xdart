"""Finite architecture and deletion census for the E3 joined route."""

from __future__ import annotations

import ast
from dataclasses import fields
import json
from pathlib import Path
import re
import subprocess

from xdart.gui.tabs.scattering.shell_values import ShellProjection


ROOT = Path(__file__).resolve().parents[3]
CENSUS_PATH = Path(__file__).with_name(
    "e3_j2_scalar_route_census_c1e354c3.json"
)


def _census() -> dict:
    return json.loads(CENSUS_PATH.read_text())


def _python_files(relative_root: str) -> tuple[Path, ...]:
    return tuple(
        sorted((ROOT / relative_root).rglob("*.py"))
    )


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text())


def _class_node(relative_path: str, name: str) -> ast.ClassDef:
    return next(
        node
        for node in _tree(relative_path).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _class_members(relative_path: str, name: str) -> set[str]:
    class_node = _class_node(relative_path, name)
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    } | {
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    } | {
        target.id
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    } | {
        node.attr
        for node in ast.walk(class_node)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _constructor_sites(name: str) -> set[str]:
    sites: set[str] = set()
    for path in _python_files("src/xdart/gui/tabs/scattering"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.Call) and _call_name(node) == name
            for node in ast.walk(tree)
        ):
            sites.add(str(path.relative_to(ROOT)))
    return sites


def test_j2_merge_keyed_scalar_route_census_is_zero() -> None:
    census = _census()
    assert census["schema"] == 1
    commit = census["baseline_commit"]
    tree = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    assert tree == census["baseline_tree"]

    for removed in census["removed_files"]:
        path = removed["path"]
        assert not (ROOT / path).exists()
        blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=ROOT,
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()
        assert blob == removed["baseline_blob"]

    files = tuple(
        path
        for root in census["scan_roots"]
        for path in _python_files(root)
    )
    for row in census["retired_tokens"]:
        token = row["token"]
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}"
            rf"(?![A-Za-z0-9_])"
        )
        hits = [
            str(path.relative_to(ROOT))
            for path in files
            if pattern.search(path.read_text())
        ]
        assert len(hits) == row["tip_expected"], (token, hits)


def test_j2_scalar_alias_members_are_absent() -> None:
    census = _census()
    violations: list[tuple[str, str, set[str]]] = []
    for row in census["forbidden_class_members"]:
        members = _class_members(row["path"], row["class"])
        overlap = members.intersection(row["members"])
        if overlap:
            violations.append((row["path"], row["class"], overlap))

    for row in census["forbidden_constructor_parameters"]:
        class_node = _class_node(row["path"], row["class"])
        constructor = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
        )
        parameters = {
            argument.arg
            for argument in (
                *constructor.args.posonlyargs,
                *constructor.args.args,
                *constructor.args.kwonlyargs,
            )
        }
        overlap = parameters.intersection(row["parameters"])
        if overlap:
            violations.append((row["path"], row["class"], overlap))

    for row in census["forbidden_module_members"]:
        module = _tree(row["path"])
        members = {
            target.id
            for statement in module.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        } | {
            statement.target.id
            for statement in module.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        overlap = members.intersection(row["members"])
        if overlap:
            violations.append((row["path"], "<module>", overlap))

    assert violations == []

    for row in census["exact_key_methods"]:
        class_node = _class_node(row["path"], row["class"])
        method = next(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == row["method"]
        )
        parameter = next(
            argument
            for argument in method.args.args
            if argument.arg == row["parameter"]
        )
        assert parameter.annotation is not None
        assert ast.unparse(parameter.annotation) == "DisplayFrameKey"

    payload = _class_node(
        "src/xdart/gui/tabs/scattering/display_values.py",
        "StandardDisplayPayload",
    )
    frame_key = next(
        node
        for node in payload.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "frame_key"
    )
    assert frame_key.value is None


def test_j2_append_reason_has_one_publication_owner() -> None:
    publication = _census()["single_publication"]
    token = publication["token"]
    owner = publication["owner"]
    definitions: list[str] = []
    forbidden_imports: list[str] = []
    production_paths = _python_files("src/xdart/gui/tabs/scattering")
    scanned_paths = production_paths + _python_files("tests/xdart/scattering")
    owner_tree = _tree(owner)
    owner_values = [
        node.value.value
        for node in ast.walk(owner_tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(
            isinstance(target, ast.Name) and target.id == token
            for target in node.targets
        )
    ]
    assert len(owner_values) == 1
    canonical_value = owner_values[0]
    literal_publications: list[str] = []
    for path in scanned_paths:
        tree = ast.parse(path.read_text())
        relative = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                )
                if any(
                    isinstance(target, ast.Name)
                    and target.id == token
                    for target in targets
                ):
                    definitions.append(relative)
            if (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == publication["forbidden_import_module"]
                    or (
                        node.level > 0
                        and node.module
                        == publication["forbidden_import_module"].rsplit(
                            ".", 1
                        )[-1]
                    )
                )
                and any(alias.name == token for alias in node.names)
            ):
                forbidden_imports.append(relative)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != token:
                        continue
                    if (
                        node.module
                        not in {
                            "output_values",
                            "xdart.gui.tabs.scattering.output_values",
                        }
                        or alias.asname is not None
                    ):
                        forbidden_imports.append(relative)
        if path in production_paths and any(
            isinstance(node, ast.Constant)
            and node.value == canonical_value
            for node in ast.walk(tree)
        ):
            literal_publications.append(relative)
    assert definitions == [owner]
    assert forbidden_imports == []
    assert literal_publications == [owner]


def test_j2_one_projection_navigation_and_mount_route() -> None:
    assert _constructor_sites("ShellProjection") == {
        "src/xdart/gui/tabs/scattering/context_projection.py"
    }
    assert _constructor_sites("BrowserProjection") == {
        "src/xdart/gui/tabs/scattering/shell_projection.py"
    }
    assert _constructor_sites("ScientificProjection") == {
        "src/xdart/gui/tabs/scattering/shell_projection.py"
    }
    assert _constructor_sites("RunStripProjection") == {
        "src/xdart/gui/tabs/scattering/run_mode_projection.py"
    }
    assert _constructor_sites("FrameNavigationProjection") == {
        "src/xdart/gui/tabs/scattering/context_runtime.py"
    }
    assert _constructor_sites("_ContextRuntime") == {
        "src/xdart/gui/tabs/scattering/context_controller.py"
    }

    projection_tree = _tree(
        "src/xdart/gui/tabs/scattering/context_projection.py"
    )
    projection_class = next(
        node
        for node in projection_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ContextProjection"
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__init__"
        for node in projection_class.body
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(projection_class)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        for node in ast.walk(projection_class)
    )
    shell_calls = [
        node
        for node in ast.walk(projection_tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "ShellProjection"
    ]
    assert len(shell_calls) == 1
    call = shell_calls[0]
    assert not any(keyword.arg is None for keyword in call.keywords)
    assert len(call.args) + len(call.keywords) == len(
        fields(ShellProjection)
    )

    page_tree = _tree("src/xdart/gui/tabs/scattering/page.py")
    calls = [
        _call_name(node)
        for node in ast.walk(page_tree)
        if isinstance(node, ast.Call)
    ]
    assert calls.count("ContextProjection") == 1
    assert calls.count("ContextController") == 1
    assert calls.count("ScatteringWorkspaceShell") == 1
    assert calls.count("build_shell") == 1
    assert calls.count("project_navigation") == 1
    assert calls.count("apply_state") == 1


def test_j2_page_and_views_have_no_second_authority_or_alias() -> None:
    forbidden = set(_census()["forbidden_page_members"])
    page_tree = _tree("src/xdart/gui/tabs/scattering/page.py")
    page_class = next(
        node
        for node in page_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ScatteringWorkspace"
    )
    class_assignments = {
        target.id
        for statement in page_class.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    } | {
        statement.target.id
        for statement in page_class.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }
    page_members = {
        node.attr
        for node in ast.walk(page_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    } | {
        node.name
        for node in ast.walk(page_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    } | class_assignments
    assert page_members.isdisjoint(forbidden)

    runtime_path = ROOT / "src/xdart/gui/tabs/scattering/context_runtime.py"
    runtime_source = runtime_path.read_text()
    assert "__all__" not in runtime_source
    for path in _python_files("src/xdart/gui/tabs/scattering"):
        if path == runtime_path:
            continue
        tree = ast.parse(path.read_text())
        assignments = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        }
        assert assignments.isdisjoint(
            {
                "_acquisition",
                "_selection",
                "_display_generation",
                "_acquisition_navigation",
                "_browse_navigation",
                "_pending_replacement",
            }
        ), path

    forbidden_imports = {
        "ContextController",
        "ContextProjection",
        "AcquisitionContext",
        "BrowseContext",
        "DisplaySelection",
        "FrameRecordStore",
        "PublicationStore",
        "RunIntentStore",
        "ScatteringCoordinator",
        "RunExecutorPort",
        "StandardRunExecutor",
        "StandardDisplayPayload",
    }
    for name in (
        "workspace_shell.py",
        "browser_view.py",
        "scientific_view.py",
        "source_view.py",
        "tools_view.py",
        "shell_widgets.py",
    ):
        tree = _tree(f"src/xdart/gui/tabs/scattering/{name}")
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported.isdisjoint(forbidden_imports), name

    budgets = {
        "context_controller.py": 500,
        "context_runtime.py": 500,
        "context_projection.py": 500,
        "shell_projection.py": 500,
        "coordinator.py": 399,
        "contracts.py": 699,
        "adapters/run_executor.py": 949,
    }
    for name, budget in budgets.items():
        assert len(
            (
                ROOT / f"src/xdart/gui/tabs/scattering/{name}"
            ).read_text().splitlines()
        ) <= budget
