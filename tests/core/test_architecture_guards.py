"""Architecture guardrails for the architecture-v2 spike."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "xrd_tools"
SRC = ROOT / "src"


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
        "xrd_tools.session",
        "xrd_tools.session.scan_session",
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


def test_first_party_never_imports_the_shim():
    """Nothing under src/ may import ssrl_xrd_tools -- the shim exists for
    USER code only."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts or path.parent.name == "ssrl_xrd_tools":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "ssrl_xrd_tools"
                       for a in node.names):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "ssrl_xrd_tools":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_core_capability_imports():
    """The load-bearing core symbols the GUI (and users) rely on."""
    from xrd_tools.io.read import relative_source_path, resolve_source_master  # noqa: F401
    from xrd_tools.io.frame_view import read_frame_view, iter_frame_views  # noqa: F401
    from xrd_tools.reduction import ReductionSession, run_reduction  # noqa: F401
    from xrd_tools.core.frame_view import FrameView, assert_frameview_equivalent  # noqa: F401


def test_wavelength_sentinel_stays_in_xdart():
    """Policy (greenfield D7): the 1.0 Å default-wavelength sentinel is an
    xdart-internal acquisition artifact.  The ONLY crossing point into the
    headless world is xdart's adapter calling the explicit
    ``allow_default_sentinel`` helpers — xrd_tools itself must never
    reference the sentinel API (None is the only missing-value sentinel
    at headless API boundaries)."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("allow_default_sentinel",
                       "DEFAULT_WAVELENGTH_SENTINEL"):
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert offenders == []
