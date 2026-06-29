# -*- coding: utf-8 -*-
"""Controls Panel V2 preview scaffold.

This widget is a thin renderer for the Qt-free
:mod:`xdart.gui.tabs.static_scan.controls_logic` profile.  It stays inside a
bounded preview container while the legacy wrangler/integration widgets remain
the production controls.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ..controls_logic import (
    AnalysisLauncherSpec,
    ControlActionSpec,
    ControlProfile,
    FieldStatus,
    SectionId,
)


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


class ActionButton(QtWidgets.QPushButton):
    """Small card action button carrying a ``ControlActionSpec``."""

    actionRequested = QtCore.Signal(object)

    def __init__(self, spec: ControlActionSpec, parent=None):
        super().__init__(spec.label, parent)
        self.setObjectName("controlsV2ActionButton")
        self._spec = spec
        self.clicked.connect(lambda: self.actionRequested.emit(self._spec.action))
        self.apply_spec(spec)

    @property
    def spec(self) -> ControlActionSpec:
        return self._spec

    def apply_spec(self, spec: ControlActionSpec) -> None:
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
        outer.setContentsMargins(7, 5, 7, 6)
        outer.setSpacing(4)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("controlsV2SectionTitle")
        outer.addWidget(self.title)
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(3)
        outer.addWidget(self.body)

    def clear_rows(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def add_row(self, widget: QtWidgets.QWidget) -> None:
        self.body_layout.addWidget(widget)


class FieldRow(QtWidgets.QWidget):
    """One typed field row rendered from ``FieldStatus``."""

    def __init__(self, status: FieldStatus, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2FieldRow")
        self._status = status
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        self.label = QtWidgets.QLabel(status.label)
        self.label.setObjectName("controlsV2FieldLabel")
        self.label.setMinimumWidth(80)
        self.value = QtWidgets.QLabel(status.value or status.reason or status.status.value)
        self.value.setObjectName("controlsV2FieldValue")
        self.value.setWordWrap(False)
        self.value.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.badge = StatusBadge(status.status.value)
        self.badge.set_status(status.status.value, status.status.value)
        self.badge.setMinimumWidth(48)
        self.badge.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lay.addWidget(self.label)
        lay.addWidget(self.value, 1)
        lay.addWidget(self.badge)
        self.setToolTip(status.reason or status.headless_key or status.session_key)

    @property
    def status(self) -> FieldStatus:
        return self._status


class ControlsPanelV2(QtWidgets.QWidget):
    """Feature-flag-ready renderer for :class:`ControlProfile`.

    It emits launcher intent only.  The owning tab decides how to open dialogs,
    run scans, or map profile changes into the legacy wrangler while V2 is
    hidden.
    """

    analysisLaunchRequested = QtCore.Signal(object)
    controlActionRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsPanelV2")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(5)

        self.summary_card = SectionCard("Run Readiness")
        self.project_card = SectionCard("Project")
        self.source_card = SectionCard("Source")
        self.experiment_card = SectionCard("Experiment")
        self.processing_card = SectionCard("Processing")
        self.output_card = SectionCard("Output")
        self.analysis_card = SectionCard("Analysis")
        lay.addWidget(self.summary_card)
        lay.addWidget(self.project_card)
        lay.addWidget(self.source_card)
        lay.addWidget(self.experiment_card)
        lay.addWidget(self.processing_card)
        lay.addWidget(self.output_card)
        lay.addWidget(self.analysis_card)
        lay.addStretch(1)
        self._profile = None

    @property
    def profile(self) -> ControlProfile | None:
        return self._profile

    def set_profile(self, profile: ControlProfile) -> None:
        self._profile = profile
        self._render_summary(profile)
        self._render_fields(profile)
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

    def _render_fields(self, profile: ControlProfile) -> None:
        sections = (
            (self.project_card, SectionId.PROJECT),
            (self.source_card, SectionId.SOURCE),
            (self.experiment_card, SectionId.EXPERIMENT),
            (self.processing_card, SectionId.PROCESSING),
            (self.output_card, SectionId.OUTPUT),
        )
        for card, section in sections:
            card.clear_rows()
            for status in profile.fields_for(section):
                card.add_row(FieldRow(status))
            self._add_actions(card, profile.actions_for(section))
        self.experiment_card.setVisible(bool(profile.show_experiment_card))
        self.processing_card.setVisible(bool(profile.show_processing_card))

    def _add_actions(
        self,
        card: SectionCard,
        actions: tuple[ControlActionSpec, ...],
    ) -> None:
        if not actions:
            return
        row = QtWidgets.QWidget()
        row.setObjectName("controlsV2ActionRow")
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(5)
        lay.addStretch(1)
        for spec in actions:
            btn = ActionButton(spec)
            btn.actionRequested.connect(self.controlActionRequested)
            lay.addWidget(btn)
        card.add_row(row)

    def _render_analysis(self, launchers: tuple[AnalysisLauncherSpec, ...]) -> None:
        self.analysis_card.clear_rows()
        for status in (self._profile.fields_for(SectionId.ANALYSIS)
                       if self._profile is not None else ()):
            self.analysis_card.add_row(FieldRow(status))
        for spec in launchers:
            btn = LauncherButton(spec)
            btn.launched.connect(self.analysisLaunchRequested)
            self.analysis_card.add_row(btn)
