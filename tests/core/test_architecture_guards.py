"""Architecture guardrails for the architecture-v2 spike."""

from __future__ import annotations

import ast
import importlib
import os, subprocess
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


def test_viewer_1d_io_is_a_pure_leaf():
    path = PACKAGE / "io" / "viewer_1d.py"; tree = ast.parse(path.read_text(), filename=str(path))
    # Finite source ratchet, not a sandbox: computed, reflective, sys.modules and native loading are review threats.
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    reserved = {"importlib", "__import__"}; imported_loaders = {*reserved, "import_module"}; type_params = tuple(getattr(ast, name) for name in ("TypeVar", "TypeVarTuple", "ParamSpec") if hasattr(ast, name))
    resolve = lambda name, level=0: ".".join((*((("xrd_tools", "io")[:max(0, 3 - level)]) if level else ()), *filter(None, name.split("."))))
    forbidden = lambda name: name == "xdart" or name.startswith("xdart.") or name == "xrd_tools.session" or name.startswith("xrd_tools.session.")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                exact = len(node.names) == 1 and alias.name == "importlib" and alias.asname is None; bound, root = alias.asname or alias.name.split(".")[0], alias.name.split(".")[0]
                if forbidden(alias.name) or root == "builtins" or root == "importlib" and not exact or bound in imported_loaders and not exact: offenders.append(f"import:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            source = resolve(node.module or "", node.level); targets = (source, *(f"{source}.{alias.name}".strip(".") for alias in node.names if alias.name != "*"))
            bindings = tuple(alias.asname or alias.name for alias in node.names)
            if any(forbidden(name) for name in targets) or source.split(".")[0] in {"builtins", "importlib"} or any(alias.name in {"import_module", "__import__"} or bound in imported_loaders for alias, bound in zip(node.names, bindings)): offenders.append(f"from:{node.lineno}")
        elif isinstance(node, ast.Call):
            function = node.func; direct = isinstance(function, ast.Name) and function.id == "__import__"
            dotted = isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "importlib" and function.attr == "import_module"
            valid = (len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
                and type(node.args[0].value) is str and bool(node.args[0].value) and not node.args[0].value.startswith(".") and not forbidden(node.args[0].value))
            if (direct or dotted) and not valid: offenders.append(f"call:{node.lineno}")
        if isinstance(node, ast.Name):
            parent = parents.get(node); grandparent = parents.get(parent)
            allowed = (node.id == "__import__" and isinstance(parent, ast.Call) and parent.func is node or node.id == "importlib" and isinstance(parent, ast.Attribute) and parent.value is node
                and parent.attr == "import_module" and isinstance(grandparent, ast.Call) and grandparent.func is parent)
            if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in reserved or isinstance(node.ctx, ast.Load) and (node.id == "__builtins__" or node.id in reserved and not allowed): offenders.append(f"name:{node.lineno}")
        binding = (node.arg if isinstance(node, ast.arg) else node.name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar, *type_params)) else node.rest
            if isinstance(node, ast.MatchMapping) else None)
        if binding in reserved: offenders.append(f"binding:{node.lineno}")
    offenders += [f"__getattr__:{node.lineno}" for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"]
    assert offenders == []


def test_viewer_1d_io_and_session_import_headlessly_in_clean_processes():
    environment = dict(os.environ, PYTHONPATH=str(SRC))
    script = """import importlib, sys
importlib.import_module(sys.argv[1])
bad = sorted(name for name in sys.modules if name.startswith(('xdart', 'PySide', 'PyQt', 'qtpy', 'napari', 'pyqtgraph', 'matplotlib')))
if sys.argv[1] == 'xrd_tools.io.viewer_1d': bad += sorted(name for name in sys.modules if name.startswith('xrd_tools.session'))
if bad: raise SystemExit(repr(bad))
"""
    for module in ("xrd_tools.io.viewer_1d", "xrd_tools.session.viewer_1d"):
        result = subprocess.run([sys.executable, "-c", script, module], env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


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


def test_static_scan_source_has_no_retired_scan_mirrors():
    """H9/8b: scan display sources must not reintroduce the retired mirrors."""
    root = SRC / "xdart" / "gui" / "tabs" / "static_scan"
    retired = ("data_1d", "data_2d", "hydrated_raw")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in retired:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert offenders == []


def test_core_capability_imports():
    """The load-bearing core symbols the GUI (and users) rely on."""
    from xrd_tools.io.read import relative_source_path, resolve_source_master  # noqa: F401
    from xrd_tools.io.frame_view import read_frame_view, iter_frame_views  # noqa: F401
    from xrd_tools.reduction import ReductionSession, run_reduction  # noqa: F401
    from xrd_tools.core.frame_view import FrameView, assert_frameview_equivalent  # noqa: F401


def test_wavelength_sentinel_has_one_headless_owner():
    """Policy (revised, X1 Slice 3a0 / R3-P1 — supersedes the greenfield-D7
    "sentinel stays in xdart" pin): the 1.0 Å default-wavelength sentinel is a
    legacy acquisition artifact whose handling now lives in EXACTLY ONE
    headless module, ``xrd_tools.core.energy`` (so the projection and the GUI
    share a single definition; ``xdart.modules.wavelength`` is a re-export
    shim).  No OTHER xrd_tools module may reference the sentinel API — the
    crossing points remain the explicit ``allow_default_sentinel`` helpers,
    and None stays the only missing-value sentinel at headless API
    boundaries."""
    owner = PACKAGE / "core" / "energy.py"
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("allow_default_sentinel",
                       "DEFAULT_WAVELENGTH_SENTINEL"):
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert offenders == []
    owner_text = owner.read_text(encoding="utf-8", errors="replace")
    assert "DEFAULT_WAVELENGTH_SENTINEL_M" in owner_text   # the one owner
