# -*- coding: utf-8 -*-
"""Hidden Controls Panel V2 scaffold.

This widget is not mounted in production yet.  It is a thin renderer for the
Qt-free :mod:`xdart.gui.tabs.static_scan.controls_logic` profile so the next
GUI pass can build typed cards behind a feature flag instead of editing the
legacy ParameterTree in place.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ..controls_logic import AnalysisLauncherSpec, ControlProfile


class StatusBadge(QtWidgets.QLabel):
    """Small status label used by card rows."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("controlsV2StatusBadge")

    def set_status(self, text: str, severity: str = "info") -> None:
        self.setText(text)
        self.setProperty("severity", severity)
        self.style().unpolish(self)
        self.style().polish(self)


class LauncherButton(QtWidgets.QPushButton):
    """Analysis launcher button carrying its launcher spec."""

    launched = QtCore.Signal(object)

    def __init__(self, spec: AnalysisLauncherSpec, parent=None):
        super().__init__(spec.label, parent)
        self.setObjectName("controlsV2LauncherButton")
        self._spec = spec
        self.clicked.connect(lambda: self.launched.emit(self._spec.tool))
        self.apply_spec(spec)

    @property
    def spec(self) -> AnalysisLauncherSpec:
        return self._spec

    def apply_spec(self, spec: AnalysisLauncherSpec) -> None:
        self._spec = spec
        self.setText(spec.label)
        self.setEnabled(bool(spec.enabled))
        self.setToolTip(spec.reason or "")
        self.setProperty("productionReady", bool(spec.production_ready))
        self.style().unpolish(self)
        self.style().polish(self)


class SectionCard(QtWidgets.QFrame):
    """A simple titled section with replaceable body rows."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2SectionCard")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 8)
        outer.setSpacing(6)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("controlsV2SectionTitle")
        outer.addWidget(self.title)
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(5)
        outer.addWidget(self.body)

    def clear_rows(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def add_row(self, widget: QtWidgets.QWidget) -> None:
        self.body_layout.addWidget(widget)


class ControlsPanelV2(QtWidgets.QWidget):
    """Feature-flag-ready renderer for :class:`ControlProfile`.

    It emits launcher intent only.  The owning tab decides how to open dialogs,
    run scans, or map profile changes into the legacy wrangler while V2 is
    hidden.
    """

    analysisLaunchRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsPanelV2")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(7)

        self.summary_card = SectionCard("Run Readiness")
        self.analysis_card = SectionCard("Analysis")
        lay.addWidget(self.summary_card)
        lay.addWidget(self.analysis_card)
        lay.addStretch(1)
        self._profile = None

    @property
    def profile(self) -> ControlProfile | None:
        return self._profile

    def set_profile(self, profile: ControlProfile) -> None:
        self._profile = profile
        self._render_summary(profile)
        self._render_analysis(profile.analysis_launchers)

    def _render_summary(self, profile: ControlProfile) -> None:
        self.summary_card.clear_rows()
        if profile.run_enabled:
            badge = StatusBadge("Ready")
            badge.set_status("Ready", "ok")
            self.summary_card.add_row(badge)
            return
        blockers = profile.run_blockers or ("No run action in this mode.",)
        for blocker in blockers:
            badge = StatusBadge(blocker)
            badge.set_status(blocker, "blocked")
            self.summary_card.add_row(badge)

    def _render_analysis(self, launchers: tuple[AnalysisLauncherSpec, ...]) -> None:
        self.analysis_card.clear_rows()
        for spec in launchers:
            btn = LauncherButton(spec)
            btn.launched.connect(self.analysisLaunchRequested)
            self.analysis_card.add_row(btn)
