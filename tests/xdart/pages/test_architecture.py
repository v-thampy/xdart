from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PAGES = ROOT / "src" / "xdart" / "gui" / "pages"
GUI_MAIN = ROOT / "src" / "xdart" / "_gui_main.py"


def test_pages_contract_import_is_lazy_and_constructs_no_qt_or_page_package():
    script = """
import sys
import xdart.gui.pages.catalog
for name in (
    'xdart.gui.tabs.static_scan',
    'xdart.gui.tabs.static_scan.static_scan_widget',
    'xdart.gui.tabs.scattering',
    'xdart.gui.tabs.scattering.page',
):
    assert name not in sys.modules, name
assert not any(name.startswith('PySide6.QtWidgets') for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_contract_modules_have_no_science_settings_or_main_imports():
    offenders = []
    forbidden_import_roots = (
        "xdart.gui.tabs", "xrd_tools", "xdart.modules", "xdart._gui_main",
    )
    for path in sorted(PAGES.glob("*.py")):
        if path.name == "legacy_static.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith(forbidden_import_roots):
                    offenders.append(f"{path.name}: imports {name}")
        text = path.read_text(encoding="utf-8")
        for token in ("QSettings", "Scattering", "Experiment", "operations_for", "OperationOwners"):
            if token in text:
                offenders.append(f"{path.name}: contains {token}")
    assert offenders == []


def test_main_host_has_no_page_name_branch_or_legacy_widget_reach_in():
    source = GUI_MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GUI_MAIN))
    string_literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "synthetic-a" not in string_literals
    assert "synthetic-b" not in string_literals
    assert "scattering-workspace" not in string_literals
    assert "staticWidget" not in source
    assert "tabs.static_scan" not in source
    assert "getattr(self.main_widget" not in source
    assert "hasattr(self.main_widget" not in source


def test_catalog_is_the_only_builtin_registration_point():
    catalog = (PAGES / "catalog.py").read_text(encoding="utf-8")
    main = GUI_MAIN.read_text(encoding="utf-8")
    assert "LEGACY_STATIC_PAGE" in catalog
    assert "BUILTIN_PAGES" in catalog
    assert "BUILTIN_PAGES" in main
    assert "PageDescriptor(" not in main
