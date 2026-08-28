"""Passive Qt rendering for accepted source-observation values."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from .contracts import SourceObservation
from .controls_readiness import SectionHeaderProjection
from .shell_widgets import (
    apply_section_header,
    source_header_projection,
)


class SourceStatusView(QtWidgets.QFrame):
    """Display-only source card; it keeps no source-selection value."""

    chooseRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("scatteringSourceStatusView")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)
        self._choose = QtWidgets.QPushButton("Choose source")
        self._choose.clicked.connect(self.chooseRequested.emit)
        layout.addWidget(self._choose)
        self._header = SectionHeaderProjection(
            "not selected",
            False,
            "Select a source to inspect it.",
        )

    def show_checking(self, selected_name: str) -> None:
        self._set_header(SectionHeaderProjection(
            "checking…",
            False,
            "Checking the selected source.",
        ))

    def show_no_source(self) -> None:
        self._set_header(SectionHeaderProjection(
            "not selected",
            False,
            "Select a source to inspect it.",
        ))

    def show_unavailable(self, selected_name: str) -> None:
        self._set_header(SectionHeaderProjection(
            "unavailable",
            False,
            "Source observation is unavailable.",
        ))

    def render(self, observation: SourceObservation) -> None:
        self._set_header(source_header_projection(observation))

    def reapply_header(self) -> None:
        """Restore the accepted observation after the form reconciles."""

        card = self._source_card()
        if card is not None:
            apply_section_header(card, self._header)

    def _set_header(self, projection: SectionHeaderProjection) -> None:
        self._header = projection
        self.reapply_header()

    def _source_card(self) -> QtWidgets.QWidget | None:
        candidate = self.parentWidget()
        while candidate is not None:
            if (
                hasattr(candidate, "set_status_text")
                and hasattr(candidate, "set_valid_marker")
                and hasattr(candidate, "status")
            ):
                return candidate
            card = getattr(candidate, "source_card", None)
            if card is not None:
                return card
            candidate = candidate.parentWidget()
        return None


__all__ = ["SourceStatusView"]
