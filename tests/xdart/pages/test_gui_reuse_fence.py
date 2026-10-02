"""The shared-component fence: page modules build no display or editor widgets of their own.

Policy: xdart_docs ``docs/design/architecture.md`` "Reusable GUI components" (maintainer, 2026-09-24;
tightened 2026-10-01, after the native RSM page's first stage built its own result view, browser menus and
Advanced dialog). A page mounts Int's owners and drives them with its own projections; where an owner falls
short, the owner is extended, with tests for every consumer. A recorded reason does not license a copy.

``_DEBT`` is the copies that existed when this fence landed (``stitch2d`` at ``3de62e92``), and, for a construct
fenced later, the copies that existed when it was fenced (FIT-TOOLS-2026-10 added tables and rectangles). Each count
is an upper bound for its file, to be lowered as the copy moves onto its owner and never raised. A new page has no
entry: its module builds none of these.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "src" / "xdart" / "gui"
#: Int's shared owners live in ``tabs/scattering``, ``widgets`` and ``themes``; everything else that
#: builds a page, a tool or an analysis window is fenced.
_OWNER_TABS = frozenset({"scattering"})

_DISPLAY_OWNER = (
    "ScientificView (tabs/scattering/scientific_view.py): mount it and drive reconcile(); extend it for a "
    "new layout, or add a shared display component beside it"
)
_EDITOR_OWNER = (
    "ControlsPanel FormRow/RangeRow (widgets/controls_panel.py) or the Advanced editor "
    "(tabs/scattering/advanced_editor.py)"
)
_BROWSER_OWNER = "BrowserView (tabs/scattering/browser_view.py)"
_BROWSE_OWNER = "file_dialogs (widgets/file_dialogs.py)"
#: Codex's combined stage-2b review F2: a Preview, a provenance record or a fit's summary is read-only text
_TEXT_OWNER = "ReadOnlyTextPane or ReadOnlyTextWindow (widgets/read_only_text.py)"
#: FIT-TOOLS-2026-10 (Codex's design review F8): a fit's result rows, and a rectangle on an image
_TABLE_OWNER = "ResultsTable (tabs/scattering/results_table.py)"
_REGION_OWNER = "ScientificView's typed rectangle (set_keyed_region, RectRegion: tabs/scattering/scientific_view.py)"

#: The constructs a page must not build, with the owner that provides them.
_CALLS: dict[str, str] = {
    **dict.fromkeys(
        (
            "PlotWidget", "PlotItem", "ImageView", "ImageItem", "GraphicsLayoutWidget", "GraphicsView",
            "HistogramLUTWidget", "HistogramLUTItem", "ColorBarItem", "PlotDataItem", "PlotCurveItem",
            "ScatterPlotItem", "InfiniteLine", "LinearRegionItem", "LegendItem", "TextItem",
            "GLViewWidget", "GLVolumeItem", "ScientificImagePane", "ImageWidget", "PlotCursorReadout",
            "IntensityControls", "getFromMatplotlib",
        ),
        _DISPLAY_OWNER,
    ),
    **dict.fromkeys(
        ("QSpinBox", "QDoubleSpinBox", "QLineEdit", "QComboBox", "QSlider", "QCheckBox"), _EDITOR_OWNER,
    ),
    "QListWidget": _BROWSER_OWNER,
    **dict.fromkeys(("QPlainTextEdit", "QTextEdit", "QTextBrowser"), _TEXT_OWNER),
    **dict.fromkeys(("QTableWidget", "QTableView"), _TABLE_OWNER),
    "RectROI": _REGION_OWNER,
}
_NAMES: dict[str, str] = {"matplotlib_colormaps": _DISPLAY_OWNER}
#: Literal markers of a copied Data Browser header and menus.
_STRINGS: dict[str, str] = dict.fromkeys(
    ("DATA BROWSER", "fileMenuButton", "configMenuButton", "helpMenuButton"), _BROWSER_OWNER,
)

_DEBT: dict[str, dict[str, int]] = {
    #: the calibration window's frame selection and frame tables, which predate the table fence (FIT-TOOLS-2026-10)
    "tabs/calibration/dialog.py": {"QTableWidget": 2},
}


def _fenced_modules() -> list[Path]:
    roots = [GUI / "pages", GUI / "tools", GUI / "analysis"]
    roots += [path for path in sorted((GUI / "tabs").iterdir())
              if path.is_dir() and path.name not in _OWNER_TABS and path.name != "__pycache__"]
    return [path for root in roots for path in sorted(root.rglob("*.py"))]


def _owner(construct: str) -> str:
    if construct.startswith("QFileDialog"):
        return _BROWSE_OWNER
    if construct.startswith("str:"):
        return _STRINGS[construct[4:]]
    return _CALLS.get(construct) or _NAMES[construct]


def _constructs(path: Path) -> Counter:
    found: Counter = Counter()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _CALLS:
                found[name] += 1
            elif name == "QFileDialog":
                found["QFileDialog"] += 1
            if isinstance(func, ast.Attribute) and isinstance(func.value, (ast.Attribute, ast.Name)):
                base = func.value.attr if isinstance(func.value, ast.Attribute) else func.value.id
                if base == "QFileDialog":
                    found[f"QFileDialog.{func.attr}"] += 1
        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in _NAMES:
                found[name] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _STRINGS:
            found[f"str:{node.value}"] += 1
    return found


def test_page_modules_build_no_display_or_editor_widgets_of_their_own():
    offenders = []
    for path in _fenced_modules():
        key = path.relative_to(GUI).as_posix()
        allowed = _DEBT.get(key, {})
        for construct, count in sorted(_constructs(path).items()):
            if count > allowed.get(construct, 0):
                offenders.append(
                    f"{key}: {construct} x{count} (debt allows {allowed.get(construct, 0)}) -> use "
                    f"{_owner(construct)}"
                )
    assert not offenders, (
        "A page module builds what a shared owner provides. Mount the owner, or extend it with tests for "
        "every consumer (architecture.md 'Reusable GUI components'); a recorded reason does not license a "
        "copy, and _DEBT is never raised:\n" + "\n".join(offenders)
    )


def test_debt_names_only_copies_that_still_exist():
    retired = []
    for key, allowed in sorted(_DEBT.items()):
        path = GUI / key
        found = _constructs(path) if path.exists() else Counter()
        retired += [f"{key}: {construct} (debt {count}, found {found[construct]})"
                    for construct, count in sorted(allowed.items()) if found[construct] == 0]
    assert not retired, "Retired copies: remove their _DEBT entries so they cannot return:\n" + "\n".join(retired)


def test_fence_covers_every_page_package_and_no_new_page_has_debt():
    fenced = {path.relative_to(GUI).as_posix() for path in _fenced_modules()}
    assert "tabs/stitch/page.py" in fenced
    assert not any(path.startswith("tabs/scattering/") for path in fenced)
    pages = {path.name for path in (GUI / "tabs").iterdir()
             if path.is_dir() and path.name not in _OWNER_TABS and path.name != "__pycache__"}
    # stitch: the fence's first debt; calibration: its two tables, fenced later (FIT-TOOLS-2026-10)
    assert {key.split("/")[1] for key in _DEBT if key.startswith("tabs/")} <= {"stitch", "calibration"} <= pages


def test_a_copied_table_or_rectangle_is_refused_with_its_owner_named(tmp_path):
    """FIT-TOOLS-2026-10 (Codex's design review F8): a page's own table or rectangle is caught, as its plots are."""
    page = tmp_path / "page.py"
    page.write_text("from pyqtgraph.Qt import QtWidgets\nimport pyqtgraph as pg\n"
                    "table = QtWidgets.QTableWidget(0, 3)\nview = QtWidgets.QTableView()\n"
                    "box = pg.RectROI((0, 0), (1, 1))\n", encoding="utf-8")
    found = _constructs(page)
    assert (found["QTableWidget"], found["QTableView"], found["RectROI"]) == (1, 1, 1)
    assert _owner("QTableWidget") == _owner("QTableView") == _TABLE_OWNER and _owner("RectROI") == _REGION_OWNER
