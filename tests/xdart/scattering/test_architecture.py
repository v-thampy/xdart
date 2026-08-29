from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import get_type_hints

import pytest

from xdart.gui.tabs.scattering import ScatteringCoordinator
from xdart.gui.tabs.scattering.contracts import (
    RunExecutorPort,
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourcePort,
)
from xdart.gui.tabs.scattering import events


PACKAGE = Path(__file__).parents[3] / "src" / "xdart" / "gui" / "tabs" / "scattering"
KERNEL_SOURCE_NAMES = {
    "contracts.py",
    "events.py",
    "state_machine.py",
    "coordinator.py",
    "start_outcomes.py",
    "start_pipeline.py",
    "controls_projection.py",
}
UI_SOURCE_NAMES = {
    "page.py",
    "workspace_shell.py",
    "browser_view.py",
    "scientific_view.py",
    "source_view.py",
    "tools_view.py",
    "shell_widgets.py",
}
ADAPTER_SOURCE = PACKAGE / "adapters" / "source.py"
RUN_EXECUTOR_SOURCE = PACKAGE / "adapters" / "run_executor.py"
OUTPUT_PREFLIGHT_SOURCE = PACKAGE / "output_preflight.py"
FORBIDDEN_MODULES = (
    "xdart.gui.tabs.static_scan_vnext",
    "xdart.gui.tabs.static_scan",
    "xdart.gui.tabs.static_scan.static_scan_widget",
    "xdart.gui.tabs.static_scan.display_frame_widget",
    "xdart.gui.tabs.static_scan.wranglers.image_wrangler",
    "xdart.gui.tabs.static_scan.wranglers.nexus_wrangler",
    "qtpy",
    "PySide6",
    "PyQt5",
    "PyQt6",
    "pyqtgraph",
    "h5py",
    "fabio",
    "typing.Generic",
    "typing.TypeVar",
)
FORBIDDEN_NAMES = {
    "BrowseLoaderPort", "ProjectionPort", "OutputPort", "OutputPlanT",
    "BrowseContextT", "ProjectionValueT", "ParameterTree", "SimpleNamespace",
    "QObject", "scan_data", "h5py", "fabio",
}
# This kernel owns no direct file I/O.  A future adapter with one of these
# method names needs an explicit reviewed boundary, not a silent exception.
PATH_METHODS = {"open", "read_text", "read_bytes", "write_text", "write_bytes"}
DIRECT_IO_CALLABLES = {"open", "builtins.open", "io.open"}


def _production_sources() -> list[Path]:
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


def _kernel_sources() -> list[Path]:
    return [PACKAGE / name for name in sorted(KERNEL_SOURCE_NAMES)]


def _forbidden_module(name: str) -> bool:
    return any(name == forbidden or name.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_MODULES)


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{module}.{alias.name}" if module else alias.name
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            targets = node.targets
            resolved = _resolve_expression(value, bindings)
            if resolved is None or not any(
                term in resolved for term in ("RunIntent", "io.", "builtins.", "typing", "pathlib")
            ):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and bindings.get(target.id) != resolved:
                    bindings[target.id] = resolved
                    changed = True
    return bindings


def _resolve_expression(node: ast.expr, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_expression(node.value, bindings)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _is_bound_name(node: ast.expr, bindings: dict[str, str], name: str) -> bool:
    resolved = _resolve_expression(node, bindings)
    return resolved == name or (resolved is not None and resolved.endswith(f".{name}"))


def _annotation_names(node: ast.AST | None, bindings: dict[str, str]) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _annotation_names(ast.parse(node.value, mode="eval").body, bindings)
    if isinstance(node, ast.Subscript):
        return _annotation_names(node.value, bindings) | _annotation_names(node.slice, bindings)
    if isinstance(node, ast.BinOp):
        return _annotation_names(node.left, bindings) | _annotation_names(node.right, bindings)
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_annotation_names(item, bindings) for item in node.elts))
    resolved = _resolve_expression(node, bindings) if isinstance(node, ast.expr) else None
    return set() if resolved is None else {resolved}


def _contains_intent_annotation(node: ast.AST | None, bindings: dict[str, str]) -> bool:
    return any(name == "RunIntent" or name.endswith(".RunIntent") for name in _annotation_names(node, bindings))


def _contains_store_annotation(node: ast.AST | None, bindings: dict[str, str]) -> bool:
    return any(name == "RunIntentStore" or name.endswith(".RunIntentStore") for name in _annotation_names(node, bindings))


def _canonical_commit_receivers(tree: ast.AST, bindings: dict[str, str]) -> set[str]:
    receivers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if _contains_store_annotation(argument.annotation, bindings):
                    receivers.add(argument.arg)
        elif isinstance(node, ast.AnnAssign) and _contains_store_annotation(node.annotation, bindings):
            resolved = _resolve_expression(node.target, bindings)
            if resolved is not None:
                receivers.add(resolved)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            resolved = _resolve_expression(value, bindings)
            is_store = (
                resolved in receivers
                or (isinstance(value, ast.Call) and _is_bound_name(value.func, bindings, "RunIntentStore"))
            )
            if not is_store:
                continue
            for target in targets:
                target_name = _resolve_expression(target, bindings)
                if isinstance(target, ast.Attribute) and target_name not in receivers:
                    receivers.add(target_name)
                    changed = True
    return receivers


def _protocol_names() -> set[str]:
    protocols: set[str] = set()
    for path in _production_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        bindings = _import_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(_is_bound_name(base, bindings, "Protocol") for base in node.bases):
                protocols.add(node.name)
    return protocols


def _is_authority_name(name: str) -> bool:
    words = set(filter(None, re.split(r"_|(?<=[a-z])(?=[A-Z])", name.lower())))
    return bool(
        words & {"intent", "source", "run"}
        and words & {"revision", "version", "epoch", "rev", "ver", "gen", "generation"}
    )


def _guard_violations(
    source: str,
    *,
    allow_reducer_candidate_return: bool = False,
    allow_reducer_candidate_transfer: bool = False,
) -> set[str]:
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    violations = set(FORBIDDEN_NAMES & (names | attrs))
    bindings = _import_bindings(tree)
    canonical_commit_receivers = _canonical_commit_receivers(tree, bindings)
    candidate_aliases = {
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if _contains_intent_annotation(argument.annotation, bindings)
    }
    persistent_containers: set[str] = set()
    freeze_aliases: set[str] = set()

    def tainted(node: ast.AST | None) -> bool:
        if isinstance(node, ast.Name):
            return node.id in candidate_aliases
        if isinstance(node, ast.NamedExpr):
            return tainted(node.value)
        if isinstance(node, ast.Starred):
            return tainted(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(tainted(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return any(tainted(item) for item in (*node.keys, *node.values))
        return False

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) or not tainted(node.value):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in candidate_aliases:
                    candidate_aliases.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr == "freeze":
            if tainted(node.value.value) or _is_bound_name(node.value.value, bindings, "RunIntent"):
                freeze_aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and isinstance(node.value, (ast.Attribute, ast.Subscript)) for target in targets):
                persistent_containers.update(target.id for target in targets if isinstance(target, ast.Name))
            if any(isinstance(target, (ast.Attribute, ast.Subscript)) for target in targets) and tainted(node.value):
                violations.add("persistent RunIntent candidate")
            if isinstance(node, ast.Assign) and any(
                (isinstance(target, ast.Name) and _is_authority_name(target.id))
                or (isinstance(target, ast.Attribute) and _is_authority_name(target.attr))
                for target in targets
            ):
                violations.add("local revision authority")
            if isinstance(node, ast.AnnAssign) and _contains_intent_annotation(node.annotation, bindings):
                if isinstance(node.target, ast.Name) and node in tree.body:
                    violations.add("module RunIntent authority")
                elif isinstance(node.target, (ast.Name, ast.Attribute)):
                    violations.add("persistent RunIntent annotation")
        if (
            isinstance(node, ast.Return)
            and tainted(node.value)
            and not allow_reducer_candidate_return
        ):
            violations.add("raw RunIntent return")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(item, ast.Name) and item.id == "property" for item in node.decorator_list
        ) and _contains_intent_annotation(node.returns, bindings):
            violations.add("persistent RunIntent annotation")
        if isinstance(node, ast.ClassDef) and any(_is_bound_name(base, bindings, "Protocol") for base in node.bases):
            if node.name not in {"SourcePort", "RunExecutorPort"}:
                violations.add(f"unexpected phase port: {node.name}")
        if isinstance(node, ast.Call):
            target = _resolve_expression(node.func, bindings)
            if target is not None and target.endswith(".RunIntentSnapshot"):
                violations.add("direct snapshot construction")
            if target is not None and target.endswith(".RunIntentSnapshot._from_owned_intent"):
                violations.add("private snapshot construction")
            if target is not None and target.endswith(".RunIntent.freeze"):
                violations.add("direct RunIntent.freeze call")
            if isinstance(node.func, ast.Name) and node.func.id in freeze_aliases:
                violations.add("direct RunIntent.freeze call")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "freeze" and tainted(node.func.value):
                violations.add("direct RunIntent.freeze call")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "extend", "add"}:
                receiver = node.func.value
                persistent = isinstance(receiver, (ast.Attribute, ast.Subscript)) or (isinstance(receiver, ast.Name) and receiver.id in persistent_containers)
                if persistent and any(tainted(argument) for argument in node.args):
                    violations.add("persistent RunIntent candidate")
            values = (*node.args, *(keyword.value for keyword in node.keywords))
            canonical_receiver = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "commit"
                and _resolve_expression(node.func.value, bindings) in canonical_commit_receivers
            )
            positional_candidate = (
                bool(node.args)
                and not isinstance(node.args[0], ast.Starred)
                and tainted(node.args[0])
            )
            keyword_candidate = any(
                keyword.arg == "candidate" and tainted(keyword.value)
                for keyword in node.keywords
            )
            direct_candidate = positional_candidate or keyword_candidate
            other_tainted = any(
                tainted(value)
                and value is not (node.args[0] if positional_candidate else None)
                and not any(keyword.arg == "candidate" and value is keyword.value for keyword in node.keywords)
                for value in values
            )
            canonical_commit = canonical_receiver and direct_candidate and not other_tainted
            if (
                any(tainted(value) for value in values)
                and not canonical_commit
                and not allow_reducer_candidate_transfer
            ):
                violations.add("raw RunIntent transfer")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_module(alias.name):
                    violations.add(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported = f"{module}.{alias.name}" if module else alias.name
                if _forbidden_module(imported) or _forbidden_module(module):
                    violations.add(f"forbidden import: {imported}")
                if imported in {"builtins.open", "io.open"}:
                    violations.add(f"direct file I/O import: {imported}")
            if node.level and (
                "static_scan" in module
                or any(alias.name == "static_scan" for alias in node.names)
            ):
                violations.add("forbidden relative legacy import")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            resolved = _resolve_expression(node, bindings)
            if resolved is not None and _forbidden_module(resolved):
                violations.add(f"forbidden reference: {resolved}")
            if resolved in DIRECT_IO_CALLABLES:
                violations.add(f"direct file I/O reference: {resolved}")

        if isinstance(node, ast.Name):
            resolved = _resolve_expression(node, bindings)
            if resolved is not None and _forbidden_module(resolved):
                violations.add(f"forbidden reference: {resolved}")
            if resolved in DIRECT_IO_CALLABLES:
                violations.add(f"direct file I/O reference: {resolved}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _resolve_expression(node.func, bindings)
        if target in DIRECT_IO_CALLABLES:
            violations.add(f"direct file I/O: {target}")
        if isinstance(node.func, ast.Attribute) and (
            node.func.attr in PATH_METHODS or node.func.attr.startswith(("read_", "write_"))
        ):
            violations.add(f"direct file I/O: {node.func.attr}")
    return violations


def test_package_import_has_no_qt_or_retired_page_dependency():
    code = """
import sys
baseline = set(sys.modules)
import xdart.gui.tabs.scattering
forbidden = (
    "qtpy", "PySide6", "PyQt5", "PyQt6", "pyqtgraph",
    "h5py", "fabio",
    "xdart.gui.tabs.static_scan",
)
leaked = sorted(name for name in set(sys.modules) - baseline if any(name == item or name.startswith(item + ".") for item in forbidden))
assert not leaked, leaked
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[3] / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_e0a_has_only_two_narrow_runtime_resolvable_phase_ready_ports():
    ports = (SourcePort, RunExecutorPort)
    assert len(ports) == 2
    assert get_type_hints(SourceCapture)
    assert get_type_hints(SourceObservationRequest)
    assert get_type_hints(SourceObservation)
    assert get_type_hints(events.PreflightAccepted)
    assert get_type_hints(events.RunIdentity.from_configuration)
    for name in events.__all__:
        value = getattr(events, name)
        if getattr(value, "__annotations__", None):
            assert get_type_hints(value), name
    for port in ports:
        assert getattr(port, "_is_protocol", False)
        for name, member in inspect.getmembers(port, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            signature = inspect.signature(member)
            hints = get_type_hints(member)
            assert signature.return_annotation is not inspect.Signature.empty
            assert "return" in hints
            for parameter in signature.parameters.values():
                if parameter.name != "self":
                    assert parameter.annotation is not inspect.Signature.empty
                    assert parameter.name in hints
                    assert parameter.kind is not parameter.VAR_KEYWORD
                    assert hints[parameter.name] is not object


def test_scattering_namespace_has_one_coordinator_and_no_legacy_alias():
    package = importlib.import_module("xdart.gui.tabs.scattering")
    coordinator_names = [name for name in package.__all__ if name.endswith("Coordinator")]

    assert coordinator_names == ["ScatteringCoordinator"]
    assert package.ScatteringCoordinator is ScatteringCoordinator
    assert "xdart.gui.tabs.static_scan_vnext" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xdart.gui.tabs.static_scan_vnext")


def test_scattering_production_has_no_obsolete_experimental_names():
    for path in _production_sources():
        source = path.read_text()
        assert "static_scan_vnext" not in source, path
        assert "StaticScanCoordinator" not in source, path


def test_e0a_production_sources_satisfy_semantic_architecture_guard():
    for path in _kernel_sources():
        assert not _guard_violations(
            path.read_text(),
            allow_reducer_candidate_return=path.name == "controls_projection.py",
            allow_reducer_candidate_transfer=path.name == "controls_projection.py",
        ), path
        tree = ast.parse(path.read_text(), filename=str(path))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
            for node in ast.walk(tree)
        ), path


def _page_ownership_violations(source: str) -> set[str]:
    """Reject the finite set of page-side model mirrors frozen for E1a."""

    forbidden = {
        "_snapshot",
        "_intent_snapshot",
        "_revision",
        "_intent_revision",
        "_generation",
        "_phase",
        "_current_source",
        "_source_spec",
        "_source_revision",
        "_source_epoch",
    }
    violations: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in forbidden
            ):
                violations.add(f"page model mirror: {target.attr}")
    return violations


def _imports_from(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    return imported


def test_e1a_exact_ui_and_adapter_allowlists_and_page_ownership():
    page = PACKAGE / "page.py"
    page_imports = _imports_from(page.read_text())
    views = tuple(
        PACKAGE / name
        for name in sorted(UI_SOURCE_NAMES - {"page.py"})
    )
    view_imports = {
        name
        for path in views
        for name in _imports_from(path.read_text())
    }
    adapter_imports = _imports_from(ADAPTER_SOURCE.read_text())
    executor_imports = _imports_from(RUN_EXECUTOR_SOURCE.read_text())

    controls_imports = {
        "xdart.gui.widgets.controls_panel.ControlsPanel",
        "xdart.gui.widgets.run_controls.RunControlsBar",
    }
    assert controls_imports <= view_imports
    assert not any(
        name.startswith("xdart.gui.tabs.static_scan")
        for name in page_imports
    )
    assert not any(
        name.startswith("xdart.gui.tabs.static_scan")
        for name in page_imports | view_imports
    )
    assert "pyqtgraph.Qt.QtCore" in page_imports
    assert "pyqtgraph.Qt.QtWidgets" in page_imports
    assert "pyqtgraph.Qt.QtCore" in view_imports
    assert "pyqtgraph.Qt.QtWidgets" in view_imports
    assert "pathlib.Path" not in view_imports
    assert not {
        violation
        for source in (path.read_text() for path in views)
        for violation in _guard_violations(source)
        if violation.startswith("direct file I/O")
    }
    assert "pathlib.Path" in adapter_imports
    assert not any(name.startswith("pyqtgraph") for name in adapter_imports)
    assert not any(name.startswith("xdart.gui.tabs.static_scan") for name in adapter_imports)
    assert not any(name.startswith("pyqtgraph") for name in executor_imports)
    assert not any(name.startswith("xdart.gui.tabs.static_scan") for name in executor_imports)
    assert "StartPipeline" not in RUN_EXECUTOR_SOURCE.read_text()
    assert not {
        violation
        for violation in _guard_violations(RUN_EXECUTOR_SOURCE.read_text())
        if violation.startswith("direct file I/O")
    }
    assert not _page_ownership_violations(page.read_text())
    assert "static_scan.static_controls_adapter" not in (
        PACKAGE / "controls_projection.py"
    ).read_text()


def test_e1a_architecture_finite_mutation_oracles():
    assert "page model mirror: _snapshot" in _page_ownership_violations(
        "class Page:\n    def __init__(self, snapshot):\n        self._snapshot = snapshot\n"
    )
    assert "page model mirror: _intent_revision" in _page_ownership_violations(
        "class Page:\n    def __init__(self):\n        self._intent_revision = 0\n"
    )
    kernel = (PACKAGE / "controls_projection.py").read_text()
    assert not _guard_violations(
        kernel,
        allow_reducer_candidate_return=True,
        allow_reducer_candidate_transfer=True,
    )
    assert "forbidden import: pyqtgraph.Qt" in _guard_violations(
        "from pyqtgraph import Qt\n"
    )
    illegal_page = _imports_from(
        "from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget\n"
    )
    assert illegal_page == {"xdart.gui.tabs.static_scan.static_scan_widget.staticWidget"}


def test_e0b_guard_allows_ephemeral_decision_input_but_rejects_intent_ownership():
    assert not _guard_violations(
        "def decide(candidate: RunIntent, store: RunIntentStore):\n    store.commit(candidate, expected_revision=0)\n",
    )
    adversaries = {
        "candidate cache": (
            "def decide(self, candidate: RunIntent):\n    self._candidate = candidate\n",
            "persistent RunIntent candidate",
        ),
        "typed field": (
            "class Pipeline:\n    _candidate: RunIntent\n",
            "persistent RunIntent annotation",
        ),
        "direct freeze": (
            "def decide(candidate: RunIntent):\n    candidate.freeze()\n",
            "direct RunIntent.freeze call",
        ),
        "module global": (
            "current: RunIntent\n",
            "module RunIntent authority",
        ),
        "container cache": (
            "def decide(self, candidate: RunIntent):\n    self._candidates.append(candidate)\n",
            "persistent RunIntent candidate",
        ),
        "raw return": (
            "def decide(candidate: RunIntent) -> RunIntent:\n    return candidate\n",
            "raw RunIntent return",
        ),
        "aliased freeze": (
            "def decide(candidate: RunIntent):\n    bound = candidate.freeze\n    bound()\n",
            "direct RunIntent.freeze call",
        ),
        "qualified global": (
            "from xrd_tools.session.run_configuration import RunIntent as Intent\ncurrent: Intent\n",
            "module RunIntent authority",
        ),
        "alias container": (
            "def decide(self, candidate: RunIntent):\n    saved = candidate\n    self._cache['intent'] = saved\n",
            "persistent RunIntent candidate",
        ),
        "snapshot constructor": (
            "from xrd_tools.session.intent_store import RunIntentSnapshot as Snapshot\nSnapshot(1, intent)\n",
            "direct snapshot construction",
        ),
        "snapshot private factory": (
            "from xrd_tools.session.intent_store import RunIntentSnapshot as Snapshot\nSnapshot._from_owned_intent(revision=1, intent=intent)\n",
            "private snapshot construction",
        ),
        "revision field": (
            "_intent_revision = 0\n",
            "local revision authority",
        ),
        "noncanonical commit": (
            "def decide(candidate: RunIntent):\n    sink.commit(candidate, expected_revision=0)\n",
            "raw RunIntent transfer",
        ),
        "canonical alias": (
            "def decide(candidate: RunIntent, store: RunIntentStore):\n    alias = store\n    alias.commit(candidate)\n",
            "raw RunIntent transfer",
        ),
        "obsolete experimental import": (
            "import xdart.gui.tabs.static_scan_vnext as legacy\n",
            "forbidden import: xdart.gui.tabs.static_scan_vnext",
        ),
    }
    for label, (source, expected) in adversaries.items():
        assert expected in _guard_violations(source), label


def test_semantic_guard_rejects_aliases_containers_authority_and_extra_ports():
    adversaries = {
        "property": (
            "class Pipeline:\n    @property\n    def current(self) -> RunIntent:\n        return self._value\n",
            "persistent RunIntent annotation",
        ),
        "container annotation": (
            "class Pipeline:\n    _history: list[RunIntent]\n",
            "persistent RunIntent annotation",
        ),
        "aliased persistent container": (
            "def decide(self, candidate: RunIntent):\n    saved = self._cache\n    saved.append(candidate)\n",
            "persistent RunIntent candidate",
        ),
        "container return": (
            "def decide(candidate: RunIntent):\n    return [candidate]\n",
            "raw RunIntent return",
        ),
        "constructor alias": (
            "from xrd_tools.session.intent_store import RunIntentSnapshot as Snapshot\nAlias = Snapshot\nAlias(1, intent)\n",
            "direct snapshot construction",
        ),
        "private constructor alias": (
            "from xrd_tools.session.intent_store import RunIntentSnapshot as Snapshot\nFactory = Snapshot._from_owned_intent\nFactory(revision=1, intent=intent)\n",
            "private snapshot construction",
        ),
        "qualified freeze": (
            "from xrd_tools.session.run_configuration import RunIntent as Intent\nIntent.freeze(candidate)\n",
            "direct RunIntent.freeze call",
        ),
        "authority fields": (
            "self._source_epoch = 1\nself._run_generation = 2\n",
            "local revision authority",
        ),
        "authority aliases": (
            "self.intent_version = 1\nself.source_generation = 2\nself.run_version = 3\n",
            "local revision authority",
        ),
        "third port": (
            "from typing import Protocol\nclass SecretPort(Protocol):\n    pass\n",
            "unexpected phase port: SecretPort",
        ),
        "relative legacy": (
            "from ..static_scan import legacy\n",
            "forbidden relative legacy import",
        ),
        "relative package legacy": (
            "from .. import static_scan\n",
            "forbidden relative legacy import",
        ),
    }
    for label, (source, expected) in adversaries.items():
        assert expected in _guard_violations(source), label
    assert _protocol_names() == {"SourcePort", "RunExecutorPort"}


def test_architecture_guard_rejects_aliases_deferred_ports_and_direct_file_io():
    mutations = {
        "old page alias": (
            "import xdart.gui.tabs.static_scan.static_scan_widget as legacy",
            "forbidden import: xdart.gui.tabs.static_scan.static_scan_widget",
        ),
        "aliased I/O": ("import h5py as storage", "forbidden import: h5py"),
        "Qt": ("from PySide6 import QtCore", "forbidden import: PySide6.QtCore"),
        "generic placeholder": ("class OutputPort: pass", "OutputPort"),
        "aliased builtin open": (
            "from builtins import open as reader",
            "direct file I/O import: builtins.open",
        ),
        "direct read": (
            "from pathlib import Path as P\nP('input').read_text()",
            "direct file I/O: read_text",
        ),
    }
    for label, (source, expected) in mutations.items():
        assert expected in _guard_violations(source), label


def test_architecture_guard_resolves_qualified_aliases_and_path_instances():
    adversaries = {
        "qualified pathlib": (
            "import pathlib as fs\nfs.Path('input').read_text()",
            "direct file I/O: read_text",
        ),
        "assigned Path": (
            "from pathlib import Path\npath = Path('input')\npath.read_bytes()",
            "direct file I/O: read_bytes",
        ),
        "qualified io": (
            "import io as streams\nstreams.open('input')",
            "direct file I/O: io.open",
        ),
        "qualified typing": (
            "import typing as t\nT = t.TypeVar('T')\nclass Deferred(t.Generic[T]): pass",
            "forbidden reference: typing.TypeVar",
        ),
    }
    for label, (source, expected) in adversaries.items():
        violations = _guard_violations(source)
        assert expected in violations, label
    assert "forbidden reference: typing.Generic" in _guard_violations(
        adversaries["qualified typing"][0],
    )


def test_architecture_guard_fails_closed_for_every_direct_io_method():
    adversaries = {
        "unbound Path.open": (
            "from pathlib import Path as P\nP.open(P('input'))",
            "direct file I/O: open",
        ),
        "annotated Path": (
            "from pathlib import Path\npath: Path = Path('input')\npath.open()",
            "direct file I/O: open",
        ),
        "attribute-held Path": (
            "from pathlib import Path\nself.path = Path('input')\nself.path.open()",
            "direct file I/O: open",
        ),
        "assigned io.open": (
            "import io\nreader = io.open\nreader('input')",
            "direct file I/O reference: io.open",
        ),
    }
    for label, (source, expected) in adversaries.items():
        assert expected in _guard_violations(source), label


def test_executor_start_failure_carries_identity_and_cleanup_status():
    signature = inspect.signature(RunExecutorPort.start)
    assert tuple(signature.parameters) == (
        "self", "configuration", "source", "run_identity", "admission",
    )
    assert "ExecutorStartFailed" in str(signature.return_annotation)


def test_e2_output_and_directory_owners_remain_on_the_exact_two_port_boundary():
    page = (PACKAGE / "page.py").read_text()
    executor = RUN_EXECUTOR_SOURCE.read_text()
    output = OUTPUT_PREFLIGHT_SOURCE.read_text()
    imports = {
        path.relative_to(PACKAGE): _imports_from(path.read_text())
        for path in _production_sources()
    }

    assert _protocol_names() == {"SourcePort", "RunExecutorPort"}
    assert "DirectoryIndexSession" not in page
    assert "h5py" not in page
    assert "NexusSink" not in page
    assert "pyqtgraph" not in output
    assert "xdart.gui.tabs.static_scan" not in output
    assert executor.count("class StandardRunExecutor") == 1
    assert "class GIRunExecutor" not in executor
    assert "class DirectoryRunExecutor" not in executor
    assert {
        path for path, names in imports.items()
        if "h5py" in names
    } == {Path("output_preflight.py")}
