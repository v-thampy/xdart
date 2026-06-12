"""Architecture guardrails for the architecture-v2 spike."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ssrl_xrd_tools"


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
        "ssrl_xrd_tools",
        "ssrl_xrd_tools.analysis.plans",
        "ssrl_xrd_tools.core",
        "ssrl_xrd_tools.core.frame_view",
        "ssrl_xrd_tools.core.metadata",
        "ssrl_xrd_tools.core.scan",
        "ssrl_xrd_tools.reduction",
        "ssrl_xrd_tools.io.nexus",
        "ssrl_xrd_tools.io.nexus_inspect",
        "ssrl_xrd_tools.sources",
        "ssrl_xrd_tools.sources.base",
        "ssrl_xrd_tools.sources.image",
        "ssrl_xrd_tools.sources.memory",
        "ssrl_xrd_tools.sources.nexus",
        "ssrl_xrd_tools.sources.registry",
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
