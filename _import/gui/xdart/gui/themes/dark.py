"""xdart dark theme (N8 / U1).

Dracula-inspired palette — soft purplish-gray background with the
signature Dracula accent colours, green/red Start/Stop buttons,
and a light-grey plot background so pyqtgraph axes/legends stay
legible against scientific imagery.

Reference palette (Dracula, https://draculatheme.com):

* ``#282a36`` — window / primary background
* ``#21222c`` — panel / menubar / statusbar background
* ``#3a3d4d`` — input / dropdown background
* ``#44475a`` — "current line", borders, selection background
* ``#f8f8f2`` — foreground (text)
* ``#6272a4`` — comment / secondary / disabled text
* ``#bd93f9`` — purple accent (focus / progress)
* ``#50fa7b`` — green (Start button)
* ``#ff5555`` — red (Stop button)
* ``#ffb86c`` — orange (hover accents)
* ``#cccccc`` — pyqtgraph plot background (light grey for readability)
* ``#1a1a1a`` — pyqtgraph plot foreground (axes / labels on the
  light plot background)
"""

from __future__ import annotations


DARK_QSS = """
/* ── Base widgets ──────────────────────────────────────────────── */
QWidget {
    background-color: #282a36;
    color: #f8f8f2;
    selection-background-color: #44475a;
    selection-color: #f8f8f2;
}

QMainWindow, QDialog {
    background-color: #282a36;
}

/* Frames + group boxes get a slightly darker panel tint so the
   layout is visually grouped without harsh borders. */
QFrame, QGroupBox {
    background-color: #21222c;
    border: 1px solid #44475a;
    border-radius: 4px;
}
QGroupBox {
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #f8f8f2;
}

/* ── Inputs ────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #3a3d4d;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 3px;
    padding: 2px 4px;
    selection-background-color: #44475a;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #bd93f9;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {
    background-color: #21222c;
    color: #6272a4;
}

QComboBox::drop-down {
    background-color: #3a3d4d;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #3a3d4d;
    color: #f8f8f2;
    selection-background-color: #44475a;
    border: 1px solid #44475a;
}

QCheckBox, QRadioButton {
    color: #f8f8f2;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    background-color: #3a3d4d;
    border: 1px solid #44475a;
    border-radius: 2px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #bd93f9;
    border: 1px solid #bd93f9;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    background-color: #21222c;
    border-color: #44475a;
}

/* ── Buttons ───────────────────────────────────────────────────── */
QPushButton {
    background-color: #3a3d4d;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 3px;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: #44475a;
    border-color: #6272a4;
}
QPushButton:pressed {
    background-color: #6272a4;
}
QPushButton:disabled {
    background-color: #21222c;
    color: #6272a4;
}

/* Accented Start / Stop — colour-coded primary CTAs.  Wranglers
   use the object names ``startButton`` and ``stopButton``. */
QPushButton#startButton {
    background-color: #50fa7b;
    color: #1a1a1a;
    font-weight: bold;
    border: 1px solid #3ddc6a;
}
QPushButton#startButton:hover {
    background-color: #6cfd92;
}
QPushButton#startButton:pressed {
    background-color: #3ddc6a;
}
QPushButton#startButton:disabled {
    background-color: #2d3d34;
    color: #6272a4;
    border-color: #2d3d34;
}

QPushButton#stopButton {
    background-color: #ff5555;
    color: #1a1a1a;
    font-weight: bold;
    border: 1px solid #e04444;
}
QPushButton#stopButton:hover {
    background-color: #ff7878;
}
QPushButton#stopButton:pressed {
    background-color: #e04444;
}
QPushButton#stopButton:disabled {
    background-color: #4a2d2d;
    color: #6272a4;
    border-color: #4a2d2d;
}

/* ── Lists / trees / tables ────────────────────────────────────── */
QListWidget, QListView, QTreeView, QTreeWidget, QTableView, QTableWidget {
    background-color: #21222c;
    color: #f8f8f2;
    border: 1px solid #44475a;
    alternate-background-color: #282a36;
    outline: 0;
}
QListWidget::item, QTreeView::item, QTableView::item {
    padding: 2px 4px;
}
QListWidget::item:selected, QTreeView::item:selected,
QTableView::item:selected {
    background-color: #44475a;
    color: #f8f8f2;
}
QListWidget::item:hover, QTreeView::item:hover,
QTableView::item:hover {
    background-color: #3a3d4d;
}

QHeaderView::section {
    background-color: #3a3d4d;
    color: #f8f8f2;
    padding: 4px;
    border: 0 solid #44475a;
    border-bottom: 1px solid #44475a;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
QTabWidget::pane {
    background-color: #282a36;
    border: 1px solid #44475a;
}
QTabBar::tab {
    background-color: #21222c;
    color: #6272a4;
    padding: 6px 12px;
    border: 1px solid #44475a;
    border-bottom: none;
}
QTabBar::tab:selected {
    background-color: #282a36;
    color: #f8f8f2;
    border-bottom: 2px solid #bd93f9;
}
QTabBar::tab:hover {
    color: #f8f8f2;
}

/* ── Menus / menubars / status bar ────────────────────────────── */
QMenuBar {
    background-color: #21222c;
    color: #f8f8f2;
    border-bottom: 1px solid #44475a;
}
QMenuBar::item:selected {
    background-color: #44475a;
}
QMenu {
    background-color: #21222c;
    color: #f8f8f2;
    border: 1px solid #44475a;
}
QMenu::item:selected {
    background-color: #44475a;
}

/* Status bar matches the menubar — no blue accent.  U1: was
   #007acc; the bottom panel reads better when it visually pairs
   with the top panel rather than fighting with the workspace. */
QStatusBar {
    background-color: #21222c;
    color: #f8f8f2;
    border-top: 1px solid #44475a;
}
QStatusBar::item {
    border: none;
}

/* ── Scrollbars ────────────────────────────────────────────────── */
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #282a36;
    border: none;
    width: 10px;
    height: 10px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #44475a;
    min-height: 20px;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:hover {
    background-color: #6272a4;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
    border: none;
}

/* ── Sliders / progress bars ───────────────────────────────────── */
QProgressBar {
    background-color: #3a3d4d;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #bd93f9;
}

/* ── Splitters ─────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #44475a;
}

/* ── Tooltips ──────────────────────────────────────────────────── */
QToolTip {
    background-color: #21222c;
    color: #f8f8f2;
    border: 1px solid #bd93f9;
    padding: 4px 6px;
}
"""


# ── pyqtgraph plot canvas: seaborn-style "darkgrid" look ─────────
#
# The surrounding GUI is dark (Dracula) but the plot canvases use
# the seaborn darkgrid aesthetic from Jupyter — lavender-grey panel,
# white gridlines, dark axis text — because (a) the user is used to
# that look from notebooks, and (b) it preserves scientific colormap
# fidelity better than a dark canvas.  Matches:
#
#     import seaborn as sns
#     sns.set_theme()                # style='darkgrid'
#     sns.set_context('talk')        # ~1.3x font sizes
#
SEABORN_BG = "#EAEAF2"      # seaborn darkgrid panel
SEABORN_FG = "#2e2e2e"      # axis / tick text
SEABORN_GRID = "#ffffff"    # gridline color
SEABORN_FONT_PT = 11        # "talk-context" rough match


def apply_dark_theme(app) -> None:
    """Apply :data:`DARK_QSS` to a :class:`QApplication` instance.

    Also seeds pyqtgraph's default plot background / foreground to
    the seaborn darkgrid colours so newly constructed plots adopt
    the look automatically.  Must be called BEFORE any pyqtgraph
    plot widgets are constructed — pyqtgraph caches the config at
    widget-creation time.

    For per-plot grid + tick-font polish, callers should also
    invoke :func:`apply_seaborn_plot_style` on each ``PlotItem``.

    Parameters
    ----------
    app
        A live :class:`QtWidgets.QApplication`.
    """
    # Build the final stylesheet by appending platform-conditional
    # overrides to the base DARK_QSS.  Currently only Windows needs
    # extra rules — Qt's default button font is larger there than
    # on macOS, so a couple of named buttons get squeezed.
    qss = DARK_QSS
    import sys as _sys
    if _sys.platform == "win32":
        qss += """
/* ── Windows-only font tweaks ─────────────────────────────────
   Calibrate + Make Mask buttons are sized for the macOS default
   button font (~13 px AppleSystemFont).  On Windows, Qt's default
   QPushButton font renders ~10% larger and the labels overflow.
   Shrink the font for just those two buttons by name.  ~10%
   reduction from the typical 9 pt Windows default → 8 pt. */
QPushButton#pyfai_calib, QPushButton#get_mask {
    font-size: 8pt;
}
"""
    app.setStyleSheet(qss)
    # Imported here to avoid a hard dependency at module import
    # time (e.g. in headless test sandboxes).
    try:
        import pyqtgraph as pg
        pg.setConfigOption("background", SEABORN_BG)
        pg.setConfigOption("foreground", SEABORN_FG)
        # Use Qt's antialiased lines so gridlines and curves look
        # like the matplotlib equivalents at typical screen scales.
        pg.setConfigOption("antialias", True)
    except ImportError:  # pragma: no cover
        pass


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
