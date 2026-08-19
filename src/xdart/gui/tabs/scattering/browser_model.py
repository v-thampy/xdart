"""Bounded values-only list model for passive E3 browser frames."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore

from .display_values import DisplayFrameKey
from .shell_widgets import frame_caption, repeated_labels


class FrameListModel(QtCore.QAbstractListModel):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._frames: tuple[DisplayFrameKey, ...] = ()
        self._captions: list[str] = []
        self._counts: dict[int, int] = {}
        self._first_rows: dict[int, int] = {}
        self._rows_by_identity: dict[int, int] = {}

    @property
    def frames(self) -> tuple[DisplayFrameKey, ...]:
        return self._frames

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frames)

    def data(self, index: QtCore.QModelIndex, role: int = 0):
        if not index.isValid() or not 0 <= index.row() < len(self._frames):
            return None
        if role in {
            int(QtCore.Qt.ItemDataRole.DisplayRole),
            int(QtCore.Qt.ItemDataRole.ToolTipRole),
        }:
            return self._captions[index.row()]
        if role == int(QtCore.Qt.ItemDataRole.UserRole):
            return self._frames[index.row()]
        return None

    def row_for(self, frame: DisplayFrameKey) -> int | None:
        return self._rows_by_identity.get(id(frame))

    def owns(self, frame: DisplayFrameKey | None) -> bool:
        if frame is None:
            return False
        row = self._rows_by_identity.get(id(frame))
        return (
            row is not None
            and 0 <= row < len(self._frames)
            and self._frames[row] is frame
        )

    def reconcile(self, frames: tuple[DisplayFrameKey, ...]) -> bool:
        if frames is self._frames:
            return False
        start = len(self._frames)
        prefix = len(frames) >= start and all(
            frame is self._frames[index]
            for index, frame in enumerate(frames[:start])
        )
        if not prefix:
            self._reset(frames)
            return True
        if len(frames) == start:
            self._frames = frames
            return False

        captions = list(self._captions)
        counts = dict(self._counts)
        first_rows = dict(self._first_rows)
        changed_old: set[int] = set()
        for row, frame in enumerate(frames[start:], start):
            label = frame.local_frame_label
            prior_count = counts.get(label, 0)
            if prior_count == 0:
                first_rows[label] = row
                captions.append(
                    frame_caption(frame, frozenset(), position=row)
                )
            else:
                first = first_rows[label]
                repeated = frozenset({label})
                captions[first] = frame_caption(
                    frames[first], repeated, position=first
                )
                captions.append(
                    frame_caption(frame, repeated, position=row)
                )
                if first < start:
                    changed_old.add(first)
            counts[label] = prior_count + 1

        self.beginInsertRows(
            QtCore.QModelIndex(),
            start,
            len(frames) - 1,
        )
        self._frames = frames
        self._captions = captions
        self._counts = counts
        self._first_rows = first_rows
        for row, frame in enumerate(frames[start:], start):
            self._rows_by_identity[id(frame)] = row
        self.endInsertRows()
        display_roles = [
            int(QtCore.Qt.ItemDataRole.DisplayRole),
            int(QtCore.Qt.ItemDataRole.ToolTipRole),
        ]
        for row in changed_old:
            index = self.index(row, 0)
            self.dataChanged.emit(index, index, display_roles)
        return True

    def _reset(self, frames: tuple[DisplayFrameKey, ...]) -> None:
        repeated = repeated_labels(frames)
        self.beginResetModel()
        self._frames = frames
        self._captions = [
            frame_caption(frame, repeated, position=row)
            for row, frame in enumerate(frames)
        ]
        self._counts = {}
        self._first_rows = {}
        self._rows_by_identity = {}
        for row, frame in enumerate(frames):
            label = frame.local_frame_label
            self._counts[label] = self._counts.get(label, 0) + 1
            self._first_rows.setdefault(label, row)
            self._rows_by_identity[id(frame)] = row
        self.endResetModel()


__all__ = ["FrameListModel"]
