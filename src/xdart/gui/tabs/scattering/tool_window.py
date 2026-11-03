"""A tool window's body, beside ``FitView``: its options in a column on the left, its displays on the right.

NEXT-128-WINDOW-POLISH lifted this from the Phase Analysis window (PHASE-ANALYSIS-128) so that the Peak Fitter mounts
the same split (the maintainer, 2026-11-02: "make it like the phase analysis window (ss2) - controls on the left, plots
on the right"). The options column is Int's controls column (``_HugWidthScrollArea``): it scrolls vertically and is
never narrower than its widest row, so the splitter shrinks the plots, never the options, and every "?" stays in view
("when it opens the left panel with the controls is not fully expanded - the question marks are hidden"). The
displays stack in a vertical splitter, and each pane scrolls on its own when the window is small.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .workspace_shell import _HugWidthScrollArea

#: the opening size (the maintainer, after the Phase Analysis mockup: "the window can be wider. 1400x1200 is also
#: fine"), within the screen, and the options column's share of the width (at least its widest row)
OPENING_SIZE = (1400, 1200)
OPTIONS_WIDTH = 440

_WIDTH_TIP = "Drag to share the width between the options and the plots."
_HEIGHT_TIP = "Drag to share the height between the fit, the trend and the table."


class ToolWindowSplit(QtWidgets.QSplitter):
    """The options (``add_card``) left of the displays (``add_display``); ``open`` sizes its window. ``name`` prefixes
    the panes' object names (``{name}Controls``, ``{name}Displays``)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, name: str) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setObjectName(f"{name}Split")
        self.setChildrenCollapsible(False)
        self.controls = _HugWidthScrollArea(self)
        self.controls.setObjectName(f"{name}Controls")
        self.controls.setWidgetResizable(True)
        column = QtWidgets.QWidget(self.controls)
        # the cards keep their natural height and the pane scrolls: a short window never squeezes a list
        column.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        self._cards = QtWidgets.QVBoxLayout(column)
        self._cards.setContentsMargins(0, 0, 6, 0)
        self._cards.addStretch(1)
        self.controls.setWidget(column)
        area = QtWidgets.QScrollArea(self)
        area.setObjectName(f"{name}Displays")
        area.setWidgetResizable(True)
        self.displays = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, area)
        self.displays.setChildrenCollapsible(False)
        area.setWidget(self.displays)
        self.addWidget(self.controls)
        self.addWidget(area)
        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 5)
        self.handle(1).setToolTip(_WIDTH_TIP)

    def add_card(self, widget: QtWidgets.QWidget) -> None:
        """A card (or row) at the end of the options column, above its closing stretch."""
        self._cards.insertWidget(self._cards.count() - 1, widget)

    def add_display(self, widget: QtWidgets.QWidget, stretch: int) -> None:
        """A display below the others, taking ``stretch`` shares of the height as the window grows."""
        self.displays.addWidget(widget)
        index = self.displays.indexOf(widget)
        self.displays.setStretchFactor(index, stretch)
        if index:
            self.displays.handle(index).setToolTip(_HEIGHT_TIP)

    def open(self, window: QtWidgets.QWidget, heights: tuple[int, ...] = ()) -> None:
        """Size ``window`` to the opening size within its screen and share it: the options their width (never below
        their widest row: the column's own minimum), the displays the rest, each display its ``heights``."""
        screen = window.screen() or QtGui.QGuiApplication.primaryScreen()
        available = (screen.availableGeometry() if screen is not None
                     else QtCore.QRect(0, 0, *OPENING_SIZE))
        width = min(OPENING_SIZE[0], int(.95 * available.width()))
        height = min(OPENING_SIZE[1], int(.95 * available.height()))
        window.resize(width, height)
        options = max(min(OPTIONS_WIDTH, width // 2), self.controls.minimumSizeHint().width())
        self.setSizes([options, max(1, window.width() - options)])
        if heights:
            self.displays.setSizes(list(heights))
