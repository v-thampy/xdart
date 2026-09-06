"""Workflow-specific analysis launchers for the E3 browser column."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.themes.spacing import current_spacing_tokens

from .external_tools import (
    ExternalToolAvailability,
    ExternalToolsProjection,
    unavailable_external_tools,
)
from .shell_values import ShellCommand, ShellCommandKind


class ToolsView(QtWidgets.QFrame):
    commandRequested = QtCore.Signal(object)

    _TOOLS = (
        ("∧ Peak Fitting", "peak_fitting"),
        ("≈ Phase Fitting", "phase_fitting"),
        ("▤ Plot Metadata", "plot_metadata"),
        ("▣ ROI Statistics", "roi_statistics"),
    )
    _EXTERNAL_VIEWERS = (
        ("◇ Open Selected in NeXpy", "nexpy_selected"),
        ("▦ silx HDF5 Viewer…", "silx_h5viewer"),
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("e3ToolsView")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tool_scroll = QtWidgets.QScrollArea()
        self.tool_scroll.setObjectName("e3ToolsScroll")
        self.tool_scroll.setWidgetResizable(True)
        self.tool_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.tool_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.tool_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.tool_scroll.setMinimumHeight(0)
        self.tool_content = QtWidgets.QWidget()
        self.tool_content.setObjectName("e3ToolsContent")
        tools_layout = QtWidgets.QVBoxLayout(self.tool_content)
        self._tools_layout = tools_layout
        self._tool_buttons: dict[str, QtWidgets.QPushButton] = {}
        for label, value in self._TOOLS:
            button = QtWidgets.QPushButton(label)
            button.setObjectName(f"e3Tool_{value}")
            button.clicked.connect(
                lambda _checked=False, target=value: self.commandRequested.emit(
                    ShellCommand(ShellCommandKind.LAUNCH_TOOL, target)
                )
            )
            tools_layout.addWidget(button)
            self._tool_buttons[value] = button
        for label, value in self._EXTERNAL_VIEWERS:
            button = QtWidgets.QPushButton(label)
            button.setObjectName(f"e3Tool_{value}")
            button.clicked.connect(
                lambda _checked=False, target=value: self.commandRequested.emit(
                    ShellCommand(
                        ShellCommandKind.LAUNCH_EXTERNAL_VIEWER, target
                    )
                )
            )
            tools_layout.addWidget(button)
            self._tool_buttons[value] = button
        self.tool_scroll.setWidget(self.tool_content)
        layout.addWidget(self.tool_scroll)
        self._natural_height: int | None = None
        self._refit_timer = QtCore.QTimer(self.tool_content)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.timeout.connect(self._refit_height)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self._apply_spacing()
        self.reconcile_external(unavailable_external_tools())

    def reconcile_external(self, projection: ExternalToolsProjection) -> None:
        if type(projection) is not ExternalToolsProjection:
            raise TypeError("external viewer projection must be exact")
        for status in projection.tools:
            self.reconcile_external_tool(status)

    def reconcile_external_tool(
        self, status: ExternalToolAvailability,
    ) -> None:
        if type(status) is not ExternalToolAvailability:
            raise TypeError("external viewer availability must be exact")
        button = self._tool_buttons.get(status.tool.value)
        if button is None:
            raise RuntimeError("external viewer button inventory changed")
        button.setEnabled(status.enabled)
        button.setToolTip(status.reason)

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        if self._natural_height is not None:
            hint.setHeight(self._natural_height)
        return hint

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if (
            event.type()
            in {QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.FontChange}
            and hasattr(self, "_tools_layout")
        ):
            self._apply_spacing()

    def _apply_spacing(self) -> None:
        tokens = current_spacing_tokens()
        self._tools_layout.setContentsMargins(
            tokens.panel_margin,
            tokens.tools_vertical_margin,
            tokens.panel_margin,
            tokens.tools_vertical_margin,
        )
        self._tools_layout.setSpacing(tokens.tools_gap)
        self._tools_layout.invalidate()
        self._refit_timer.start(0)

    def _refit_height(self) -> None:
        content_height = self.tool_content.sizeHint().height()
        self.tool_content.setMinimumHeight(content_height)
        # Account for both the scroll viewport chrome and this outer QFrame.
        # The old maximum used only ``content_height`` and therefore allocated
        # four pixels less than the content actually needed under the app QSS;
        # Qt correctly introduced an otherwise needless inner scrollbar.
        viewport_chrome = max(
            0,
            self.tool_scroll.height()
            - self.tool_scroll.viewport().height(),
        )
        outer_chrome = max(
            0,
            self.height() - self.contentsRect().height(),
        )
        self._natural_height = (
            content_height + viewport_chrome + outer_chrome
        )
        self.setMaximumHeight(self._natural_height)
        self.updateGeometry()


__all__ = ["ToolsView"]
