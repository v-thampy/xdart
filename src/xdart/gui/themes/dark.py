"""xdart UI theme — token-based Dark + Light palettes (Direction A).

The QSS is a single ``_QSS_TEMPLATE`` of ``$token`` placeholders
(``string.Template`` -- ``$`` avoids clashing with QSS ``{ }`` blocks).
``DARK``/``LIGHT`` (from the Direction-A mockup; XDART_REDESIGN.md s3)
supply the values; ``render_qss`` substitutes and ``apply_theme`` re-applies
on a switch.  ``DARK`` refines the original Dracula palette (same #bd93f9
accent / #5269a8 browse).  ``apply_dark_theme``/``DARK_QSS`` stay as aliases.
"""

from __future__ import annotations

import string

DARK = {
    "accent": "#bd93f9",
    "accent_text": "#d6c2fb",
    "border": "#2b2e3b",
    "browse": "#5269a8",
    "browse_border": "#6075b5",
    "browse_text": "#f8f8f2",
    "canvas": "#15161d",
    "card": "#191b23",
    "field": "#2d3040",
    "field_border": "#3c4052",
    "gap": "#cfd4e3",
    "grid": "#d6d9e0",
    "menubar": "#1b1d27",
    "panel": "#1e2029",
    "plot1d_bg": "#eceef2",
    "plot2d_bg": "#0f1118",
    "row_active_text": "#e7defc",
    "start": "#46c98a",
    "start_text": "#0c1f16",
    "stop_bg": "#3a2730",
    "stop_border": "#5c3a44",
    "stop_text": "#e08597",
    "text": "#eef0f7",
    "text_2": "#cfd4e3",
    "text_3": "#9aa0b5",
    "text_muted": "#828799",
    "titlebar": "#16171f",
    "tree_band": "#52566d",
    "win_bg": "#21232e",
}

LIGHT = {
    "accent": "#7c5cff",
    "accent_text": "#5a3fd6",
    "border": "#e1e4ec",
    "browse": "#3a6fd6",
    "browse_border": "#3a6fd6",
    "browse_text": "#ffffff",
    "canvas": "#dcdee5",
    "card": "#ffffff",
    "field": "#eef0f5",
    "field_border": "#d7dbe6",
    "gap": "#ffffff",
    "grid": "#e8eaf0",
    "menubar": "#eef0f5",
    "panel": "#f5f6fa",
    "plot1d_bg": "#fbfbfd",
    "plot2d_bg": "#ffffff",
    "row_active_text": "#3a2f6b",
    "start": "#1f9d57",
    "start_text": "#ffffff",
    "stop_bg": "#ffffff",
    "stop_border": "#e3b9c0",
    "stop_text": "#c0392b",
    "text": "#1b2030",
    "text_2": "#2b3043",
    "text_3": "#5a6075",
    "text_muted": "#8a90a2",
    "titlebar": "#e9ebf1",
    "tree_band": "#dfe3ee",
    "win_bg": "#ffffff",
}

def _hex(c):
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _clamp(x):
    return max(0, min(255, int(round(x))))


def _shade(c, pct):
    """Lighten (pct>0, toward white) or darken (pct<0, toward black) a hex."""
    r, g, b = _hex(c)
    if pct >= 0:
        r, g, b = (v + (255 - v) * pct for v in (r, g, b))
    else:
        r, g, b = (v * (1 + pct) for v in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(_clamp(r), _clamp(g), _clamp(b))


def _blend(a, b, t):
    """Linear interpolate hex ``a``->``b`` by ``t`` in [0, 1]."""
    ra, ga, ba = _hex(a)
    rb, gb, bb = _hex(b)
    return "#{:02x}{:02x}{:02x}".format(
        _clamp(ra + (rb - ra) * t), _clamp(ga + (gb - ga) * t),
        _clamp(ba + (bb - ba) * t))


_ACTIVE = {"active": "#ffb86c", "active_border": "#e0a050",
           "active_hover": "#ffc987", "active_muted": "#5a4a35"}


def _resolve(base, *, is_light):
    """Palette + the derived state shades the template needs."""
    p = dict(base)
    p["accent_on_text"] = "#ffffff" if is_light else "#1a1a1a"
    p["accent_hover"] = _shade(base["accent"], 0.12)
    p["accent_muted"] = _blend(base["accent"], base["field"], 0.6)
    p["browse_hover"] = _shade(base["browse"], 0.14)
    p["browse_pressed"] = _shade(base["browse"], -0.16)
    p["browse_muted"] = _blend(base["browse"], base["field"], 0.6)
    p["tree_band_disabled"] = _blend(base["tree_band"], base["panel"], 0.5)
    p["start_border"] = _shade(base["start"], -0.16)
    p["start_hover"] = _shade(base["start"], 0.14)
    p["start_muted"] = _blend(base["start"], base["field"], 0.6)
    p["stop_hover"] = _blend(base["stop_bg"], base["stop_text"], 0.15)
    p["stop_muted"] = _blend(base["stop_bg"], base["panel"], 0.5)
    p.update(_ACTIVE)
    return p


_THEMES = {"dark": (DARK, False), "light": (LIGHT, True)}

_QSS_TEMPLATE = """
/* ── Base widgets ──────────────────────────────────────────────── */
QWidget {
    background-color: $win_bg;
    color: $text;
    selection-background-color: $field_border;
    selection-color: $text;
}

/* Disabled text reads as dimmed everywhere (D2): the input widgets already
   grey out (QLineEdit/QComboBox/QPushButton :disabled below), but a disabled
   panel's *labels* and check/radio text stayed bright, so a disabled 2-D /
   integration panel didn't read as off.  Grey the labels too. */
QLabel:disabled,
QCheckBox:disabled,
QRadioButton:disabled,
QGroupBox:disabled,
QGroupBox::title:disabled {
    color: $text_muted;
}

QMainWindow, QDialog {
    background-color: $win_bg;
}

/* Frames + group boxes get a slightly darker panel tint so the
   layout is visually grouped without harsh borders. */
QFrame, QGroupBox {
    background-color: $panel;
    border: 1px solid $field_border;
    border-radius: 8px;
}
/* Display top bar: ONE continuous bar -- the cluster/title containers and
   the title label are borderless + transparent so only frame_top's panel
   shows (the buttons inside keep their own styling).  setFrameShape can't
   override QSS, so the exception lives here. */
QFrame#frame_4, QFrame#frame_5, QFrame#frame_6, QLabel#labelCurrent {
    border: none;
    background: transparent;
}
/* DATA BROWSER header bar: a clean title row (label left, Refresh right) — no
   card box around it. */
QFrame#dataBrowserBar {
    border: none;
    background: transparent;
}
QLabel#dataBrowserHeader {
    font-weight: 600;
    color: $text_muted;
    padding-left: 2px;
}
QGroupBox {
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: $text;
}

/* ── Inputs ────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: $field;
    color: $text;
    border: 1px solid $field_border;
    border-radius: 5px;
    padding: 2px 4px;
    selection-background-color: $field_border;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid $accent;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {
    background-color: $panel;
    color: $text_muted;
}

/* Inline path field + Browse button (StrBrowseParameterItem — the str_browse
   wrangler rows: Project Folder, Calibration, Image/Mask File, Save Path).
   Object-named so they theme through QSS and recolour on a live theme switch,
   replacing the old inline Dracula hex that broke under the Light palette. */
QLineEdit#BrowsePathEdit {
    background-color: $field;
    border: 1px solid $field_border;
}
QLineEdit#BrowsePathEdit:focus {
    border: 1px solid $accent;
}
QPushButton#BrowseButton {
    background-color: $browse;
    color: $browse_text;
    border: 1px solid $browse_border;
    border-radius: 6px;
    padding: 1px 6px;
}
QPushButton#BrowseButton:hover {
    background-color: $browse_hover;
    border-color: $accent;
}
QPushButton#BrowseButton:pressed {
    background-color: $browse_pressed;
}
QPushButton#BrowseButton:disabled {
    background-color: $browse_muted;
    color: $text_muted;
    border-color: $browse_muted;
}

QComboBox::drop-down {
    background-color: $field;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: $field;
    color: $text;
    selection-background-color: $field_border;
    border: 1px solid $field_border;
}

QCheckBox, QRadioButton {
    color: $text;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    background-color: $field;
    border: 1px solid $field_border;
    border-radius: 2px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: $accent;
    border: 1px solid $accent;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    background-color: $panel;
    border-color: $field_border;
}

/* ── Buttons ───────────────────────────────────────────────────── */
QPushButton {
    background-color: $field;
    color: $text;
    border: 1px solid $field_border;
    border-radius: 6px;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: $field_border;
    border-color: $text_muted;
}
QPushButton:pressed {
    background-color: $text_muted;
}
QPushButton:disabled {
    background-color: $panel;
    color: $text_muted;
}

/* Checkable *toggle* buttons (Live, Batch, Auto, Legend, Share Axis,
   X Range, …) — fill with the purple accent when active so the on/off
   state is obvious.  ``:checked`` only matches checkable buttons, so
   plain action buttons are unaffected. */
QPushButton:checked {
    background-color: $accent;
    color: $accent_on_text;
    border: 1px solid $accent;
    font-weight: bold;
}
QPushButton:checked:hover {
    background-color: $accent_hover;
}
QPushButton:checked:disabled {
    background-color: $accent_muted;
    color: $text_muted;
    border-color: $accent_muted;
}

/* Accented Start / Stop — colour-coded primary CTAs.  Wranglers
   use the object names ``startButton`` and ``stopButton``. */
QPushButton#startButton {
    background-color: $start;
    color: $start_text;
    font-weight: bold;
    border: 1px solid $start_border;
}
QPushButton#startButton:hover {
    background-color: $start_hover;
}
QPushButton#startButton:pressed {
    background-color: $start_border;
}
QPushButton#startButton:disabled {
    background-color: $start_muted;
    color: $text_muted;
    border-color: $start_muted;
}

/* Phase B: the single action button morphs to ORANGE while a run is active
   (Pause / Resume), keyed on a dynamic ``runPhase`` property the wrangler sets
   (idle/unset -> the green rules above; "active" -> orange here).  The attribute
   selector outranks the plain #startButton rules, and these come later, so the
   orange state wins while active.  Live/Batch keep their purple :checked styling
   (those are checkable toggles; startButton is not, so no collision). */
QPushButton#startButton[runPhase="active"] {
    background-color: $active;
    color: $start_text;
    font-weight: bold;
    border: 1px solid $active_border;
}
QPushButton#startButton[runPhase="active"]:hover {
    background-color: $active_hover;
}
QPushButton#startButton[runPhase="active"]:pressed {
    background-color: $active_border;
}
QPushButton#startButton[runPhase="active"]:disabled {
    background-color: $active_muted;
    color: $text_muted;
    border-color: $active_muted;
}

QPushButton#stopButton {
    background-color: $stop_bg;
    color: $stop_text;
    font-weight: bold;
    border: 1px solid $stop_border;
}
QPushButton#stopButton:hover {
    background-color: $stop_hover;
}
QPushButton#stopButton:pressed {
    background-color: $stop_border;
}
QPushButton#stopButton:disabled {
    background-color: $stop_muted;
    color: $text_muted;
    border-color: $stop_muted;
}

/* ── Lists / trees / tables ────────────────────────────────────── */
QListWidget, QListView, QTreeView, QTreeWidget, QTableView, QTableWidget {
    background-color: $panel;
    color: $text;
    border: 1px solid $field_border;
    alternate-background-color: $win_bg;
    outline: 0;
}
QListWidget::item, QTreeView::item, QTableView::item {
    padding: 2px 4px;
}
QListWidget::item:selected, QTreeView::item:selected,
QTableView::item:selected {
    background-color: $field_border;
    color: $text;
}
QListWidget::item:hover, QTreeView::item:hover,
QTableView::item:hover {
    background-color: $field;
}
/* Disabled item views (e.g. the wrangler ParameterTree during a run) must dim
   their text like the QLabel/QCheckBox/QPushButton :disabled rules above —
   without this the parameter-name labels stay bright and the panel reads as
   active even though it's locked.  Group-header rows that carry a widget-local
   colour (image_wrangler.stylize_ParameterTree) get their own :disabled variant
   there. */
QListWidget:disabled, QListView:disabled, QTreeView:disabled,
QTreeWidget:disabled, QTableView:disabled, QTableWidget:disabled {
    color: $text_muted;
}
QListWidget::item:disabled, QTreeView::item:disabled,
QTableView::item:disabled {
    color: $text_muted;
}

QHeaderView::section {
    background-color: $field;
    color: $text;
    padding: 4px;
    border: 0 solid $field_border;
    border-bottom: 1px solid $field_border;
}

/* Wrangler ParameterTree (image + nexus) — Stage 3a card grouping.  The
   top-level group rows (Project / Calibration / Data / Background / Output …)
   render as uppercase card-header BANDS; the editable fields pick up the
   themed field tint.  Object-named (setObjectName('WranglerTree')) so this
   themes in both Dark AND Light and live-switches, replacing the old
   widget-local Dracula stylesheet that broke under the Light palette. */
QTreeView#WranglerTree {
    background-color: $card;
    alternate-background-color: $card;
}
QTreeView#WranglerTree::item:has-children {
    background-color: $tree_band;
    color: $text;
    font-weight: 700;
}
QTreeView#WranglerTree::item:has-children:disabled {
    background-color: $tree_band_disabled;
    color: $text_muted;
}
QTreeView#WranglerTree QLineEdit,
QTreeView#WranglerTree QComboBox,
QTreeView#WranglerTree QAbstractSpinBox {
    background-color: $field;
    color: $text;
}

/* ── Integration panel sub-cards (Stage 3b) ───────────────────── */
/* The 1-D / 2-D blocks become distinct sub-cards (radius 6, $card body);
   their inner header/range frames go transparent so only the outer card
   border shows (the generated UI gave every nested QFrame a border).  The
   "1-D" / "2-D" tags become accent pills (were a hardcoded grey rgba), and
   the numeric range/Pts fields pick up a monospace face. */
QFrame#frame1D, QFrame#frame2D {
    background-color: $card;
    border: 1px solid $field_border;
    border-radius: 6px;
}
QFrame#frame1D_header, QFrame#frame1D_range,
QFrame#frame2D_header, QFrame#frame2D_range {
    border: none;
    background-color: transparent;
}
QLabel#label1D, QLabel#label2D {
    background-color: $accent;
    color: $accent_on_text;
    border-radius: 4px;
    font-weight: 700;
}
/* When the 1-D/2-D card is disabled (viewer modes / Int-1D 2-D block / during a
   run) the pill must dim with the rest of the card — the ID selector outranks
   the generic QLabel:disabled rule, so it needs its own :disabled variant. */
QLabel#label1D:disabled, QLabel#label2D:disabled {
    background-color: $accent_muted;
    color: $text_muted;
}

/* ── Tools placeholder + Metadata popup (Stage 4) ─────────────── */
/* The inline metadata table moved into the on-demand "Metadata" dialog; the
   freed bottom-left corner is a dashed Tools card reserving space for planned
   modules.  All direct #id selectors (metaFrame is not reparented), so the
   chip's mono face is safe here. */
QLabel#toolsHeader {
    color: $text_muted;
    font-weight: 700;
}
QFrame#toolsPlaceholder {
    background-color: $card;
    border: 1px dashed $field_border;
    border-radius: 7px;
}
QFrame#toolDot {
    background-color: $accent;
    border: none;
    border-radius: 3px;
}
QLabel#toolLabel {
    color: $text_2;
    font-weight: 600;
}
QLabel#toolChip {
    background-color: $accent_muted;
    color: $accent_text;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
}
QLabel#toolsNote {
    color: $text_muted;
}
/* Active-tool "Open" affordance + the Peak Fitting popup's primary Fit button. */
QPushButton#toolOpen, QPushButton#peakFitGo {
    background-color: $accent;
    color: $accent_on_text;
    border: 1px solid $accent;
    border-radius: 4px;
    padding: 1px 10px;
    font-weight: 700;
}
QPushButton#toolOpen:hover, QPushButton#peakFitGo:hover {
    background-color: $accent_hover;
    border-color: $accent_hover;
}
QLabel#peakFitStatus {
    color: $text_muted;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
QTabWidget::pane {
    background-color: $win_bg;
    border: 1px solid $field_border;
}
QTabBar::tab {
    background-color: $panel;
    color: $text_muted;
    padding: 6px 12px;
    border: 1px solid $field_border;
    border-bottom: none;
}
QTabBar::tab:selected {
    background-color: $win_bg;
    color: $text;
    border-bottom: 2px solid $accent;
}
QTabBar::tab:hover {
    color: $text;
}

/* ── Menus / menubars / status bar ────────────────────────────── */
QMenuBar {
    background-color: $panel;
    color: $text;
    border-bottom: 1px solid $field_border;
}
QMenuBar::item:selected {
    background-color: $field_border;
}
QMenu {
    background-color: $panel;
    color: $text;
    border: 1px solid $field_border;
}
QMenu::item:selected {
    background-color: $field_border;
}

/* Status bar matches the menubar — no blue accent.  U1: was
   #007acc; the bottom panel reads better when it visually pairs
   with the top panel rather than fighting with the workspace. */
QStatusBar {
    background-color: $panel;
    color: $text;
    border-top: 1px solid $field_border;
}
QStatusBar::item {
    border: none;
}

/* ── Scrollbars ────────────────────────────────────────────────── */
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: $win_bg;
    border: none;
    width: 10px;
    height: 10px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: $field_border;
    min-height: 20px;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:hover {
    background-color: $text_muted;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
    border: none;
}

/* ── Sliders / progress bars ───────────────────────────────────── */
QProgressBar {
    background-color: $field;
    color: $text;
    border: 1px solid $field_border;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: $accent;
}

/* ── Splitters ─────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: $field_border;
}

/* ── Tooltips ──────────────────────────────────────────────────── */
QToolTip {
    background-color: $panel;
    color: $text;
    border: 1px solid $accent;
    padding: 4px 6px;
}"""


def render_qss(name="dark"):
    """Render the QSS for theme ``name`` ("dark"/"light")."""
    base, is_light = _THEMES.get(name, _THEMES["dark"])
    return string.Template(_QSS_TEMPLATE).substitute(
        _resolve(base, is_light=is_light))


def apply_theme(app, name="dark") -> None:
    """Apply theme ``name`` to a live QApplication (QSS + pyqtgraph config).

    Must run before pyqtgraph plot widgets are constructed."""
    qss = render_qss(name)
    import sys as _sys
    if _sys.platform == "win32":
        qss += ("\nQPushButton#pyfai_calib, QPushButton#get_mask "
                "{ font-size: 8.5pt; }\n")
    app.setStyleSheet(qss)
    try:
        import pyqtgraph as pg
        pg.setConfigOption("background", SEABORN_BG)
        pg.setConfigOption("foreground", SEABORN_FG)
        pg.setConfigOption("antialias", True)
    except ImportError:  # pragma: no cover
        pass


def apply_dark_theme(app) -> None:
    """Back-compat: apply the dark theme."""
    apply_theme(app, "dark")


#: Back-compat string export (the rendered dark QSS).
DARK_QSS = render_qss("dark")


SEABORN_BG = "#EAEAF2"      # seaborn darkgrid panel
SEABORN_FG = "#2e2e2e"      # axis / tick text
SEABORN_GRID = "#ffffff"    # gridline color
SEABORN_FONT_PT = 11        # "talk-context" rough match


def apply_seaborn_plot_style(plot, *, font_pt: int = SEABORN_FONT_PT,
                              grid: bool = True) -> None:
    """Apply seaborn-darkgrid + talk-context styling to a PlotItem.

    Mirrors what :func:`seaborn.set_theme` + :func:`seaborn.set_context`
    do for matplotlib axes — gridlines on with a soft alpha,
    slightly larger label / tick fonts, sans-serif by default.

    Idempotent: safe to call multiple times.  Callers do this once
    per :class:`pyqtgraph.PlotItem` after it's been added to a
    layout.  For image plots, pass ``grid=False`` — gridlines on
    top of a viridis colormap clutter the view and don't match
    seaborn's own ``imshow`` defaults either.

    Parameters
    ----------
    plot
        A :class:`pyqtgraph.PlotItem` (returned by ``addPlot()``).
    font_pt
        Point size for axis labels and tick labels.  Default 11
        matches seaborn's "talk" context roughly.
    grid
        Whether to enable gridlines.  True for 1D / line plots,
        False for image / heatmap plots.
    """
    try:
        import pyqtgraph as pg  # noqa: F401
        from PySide6.QtGui import QFont
    except ImportError:  # pragma: no cover
        return

    if grid:
        # alpha gives the soft seaborn look — white gridlines
        # bleed through the lavender-grey panel without
        # overwhelming the data traces.  0.25 is a hair under
        # half of the earlier 0.6 default; 0.3 still read as
        # too prominent on a high-DPI display.
        try:
            plot.showGrid(x=True, y=True, alpha=0.25)
        except Exception:  # pragma: no cover — older pyqtgraph
            pass

    # Tick + label fonts.  pyqtgraph's AxisItem stores the tick
    # font separately from the label font; set both so "talk
    # context" applies consistently.
    font = QFont()
    font.setPointSize(font_pt)
    for axis_name in ("bottom", "left", "top", "right"):
        try:
            axis = plot.getAxis(axis_name)
        except Exception:
            continue
        if axis is None:
            continue
        try:
            axis.setStyle(tickFont=font)
        except Exception:
            pass
        if axis.label is not None:
            try:
                axis.label.setFont(font)
            except Exception:
                pass
