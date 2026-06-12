"""Architecture guardrails for the architecture-v2 spike."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "xrd_tools"


def _python_files():
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" not in path.parts:
            yield path


def test_ssrl_tree_does_not_import_xdart():
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "xdart" or alias.name.startswith("xdart."):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "xdart" or module.startswith("xdart."):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_headless_contract_imports_do_not_pull_gui_modules():
    before = set(sys.modules)
    for name in (
        "xrd_tools",
        "xrd_tools.analysis.plans",
        "xrd_tools.core",
        "xrd_tools.core.frame_view",
        "xrd_tools.core.metadata",
        "xrd_tools.core.scan",
        "xrd_tools.reduction",
        "xrd_tools.io.nexus",
        "xrd_tools.io.nexus_inspect",
        "xrd_tools.sources",
        "xrd_tools.sources.base",
        "xrd_tools.sources.image",
        "xrd_tools.sources.memory",
        "xrd_tools.sources.nexus",
        "xrd_tools.sources.registry",
    ):
        importlib.import_module(name)
    loaded = set(sys.modules) - before
    forbidden_roots = (
        "PySide",
        "PyQt",
        "qtpy",
        "napari",
        "pyqtgraph",
        "matplotlib",
    )
    forbidden = sorted(
        name for name in loaded
        if name.startswith(forbidden_roots)
    )
    assert forbidden == []
