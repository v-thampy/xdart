"""Passive Data Browser rendering for the E3 visual shell."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.themes.spacing import current_spacing_tokens
from xdart.utils.throttle import Coalescer

from .browser_model import FrameListModel
from .display_values import DisplayFrameKey
from .shell_values import (
    BrowserProjection,
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)


_USER_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_ACCUMULATING_PLOT_MODES = frozenset(
    {"Overlay", "Waterfall", "Sum", "Average"}
)
_VISIT_ACCUMULATING_PLOT_MODES = frozenset({"Overlay", "Waterfall"})
_FRAME_SELECTION_SETTLE_MS = 100
_FRAME_NAVIGATION_KEYS = frozenset(
    {
        int(QtCore.Qt.Key.Key_Up),
        int(QtCore.Qt.Key.Key_Down),
        int(QtCore.Qt.Key.Key_PageUp),
        int(QtCore.Qt.Key.Key_PageDown),
        int(QtCore.Qt.Key.Key_Home),
        int(QtCore.Qt.Key.Key_End),
    }
)


class _AccumulatingFrameClickFilter(QtCore.QObject):
    """Own accumulating click intent separately from trace membership.

    The production browser keeps ``ExtendedSelection`` active and owns plain
    mouse clicks explicitly.  Overlay/Waterfall visits replace visible
    membership; Ctrl/Meta toggles and Shift ranges remain explicit gestures,
    while Sum/Average retain plain-click toggling and native keyboard handling.
    """

    def __init__(
        self,
        view: QtWidgets.QListView,
        get_plot_mode,
        set_selection_intent,
    ) -> None:
        super().__init__(view)
        self._view = view
        self._get_plot_mode = get_plot_mode
        self._set_selection_intent = set_selection_intent

    def eventFilter(self, watched, event) -> bool:
        # The filter is installed only on the viewport.  Check the event
        # before dereferencing the view so DeferredDelete teardown events
        # cannot reach an already-destroyed QListView wrapper.
        if event.type() != QtCore.QEvent.Type.MouseButtonPress:
            return False
        try:
            button = event.button()
            modifiers = event.modifiers()
        except (AttributeError, RuntimeError):
            return False
        if button != QtCore.Qt.MouseButton.LeftButton:
            return False
        shift_modifier = bool(
            modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        toggle_modifier = bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
        )
        plot_mode = self._get_plot_mode()
        accumulating = plot_mode in _ACCUMULATING_PLOT_MODES
        visit_mode = plot_mode in _VISIT_ACCUMULATING_PLOT_MODES
        if not toggle_modifier and not accumulating:
            return False
        try:
            position = event.position().toPoint()
        except AttributeError:
            position = event.pos()
        index = self._view.indexAt(position)
        if not index.isValid():
            # An empty-area click must not destroy an accumulating selection.
            return accumulating
        selection = self._view.selectionModel()
        explicit = shift_modifier or toggle_modifier or not visit_mode
        self._set_selection_intent(explicit, index.data(_USER_ROLE))
        anchor = selection.currentIndex()
        selection.setCurrentIndex(
            index,
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        selection.select(
            (
                QtCore.QItemSelection(
                    *sorted((anchor, index), key=lambda item: item.row())
                )
                if shift_modifier and anchor.isValid()
                else index
            ),
            (
                QtCore.QItemSelectionModel.SelectionFlag.Toggle
                if explicit and not shift_modifier
                else QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        self._set_selection_intent(None, None)
        return True


class _FrameSelectionCadenceFilter(QtCore.QObject):
    """Bracket a held navigation key as one deferred browse gesture."""

    def __init__(self, owner: "BrowserView") -> None:
        super().__init__(owner.frames)
        self._owner = owner

    def eventFilter(self, watched, event) -> bool:
        event_type = event.type()
        if event_type not in {
            QtCore.QEvent.Type.KeyPress,
            QtCore.QEvent.Type.KeyRelease,
        }:
            return False
        try:
            key = int(event.key())
            auto_repeat = bool(event.isAutoRepeat())
            modifiers = event.modifiers()
        except (AttributeError, RuntimeError):
            return False

        try:
            owner = self._owner
            if event_type == QtCore.QEvent.Type.KeyPress:
                if key in _FRAME_NAVIGATION_KEYS:
                    if not owner._frame_gesture_active:
                        owner._begin_frame_gesture(modifiers)
                    return False
                return False

            if (
                owner._frame_gesture_active
                and not auto_repeat
                and (
                    key in _FRAME_NAVIGATION_KEYS
                    or key == int(QtCore.Qt.Key.Key_Shift)
                )
            ):
                owner._finish_frame_gesture()
        except RuntimeError:
            # DeferredDelete may invalidate the owning widget before a queued
            # key event reaches the application event filter.
            return False
        return False


class _ElidedDirectoryLabel(QtWidgets.QLabel):
    _MAX_GLYPHS = 30

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = ""
        self.setObjectName("e3BrowserDirectory")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

    def set_path(self, path: str) -> None:
        self._path = path
        self.setToolTip(path)
        self._refresh_text()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_text()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._refresh_text()

    def _refresh_text(self) -> None:
        if not self._path:
            self.setText("")
            return
        metrics = self.fontMetrics()
        glyph_width = metrics.horizontalAdvance("0" * self._MAX_GLYPHS)
        bracket_width = metrics.horizontalAdvance("[]")
        available = (
            self.width() - bracket_width
            if self.width() > bracket_width
            else glyph_width
        )
        text = metrics.elidedText(
            self._path,
            QtCore.Qt.TextElideMode.ElideMiddle,
            min(glyph_width, available),
        )
        if len(text) > self._MAX_GLYPHS:
            left = (self._MAX_GLYPHS - 1) // 2
            right = self._MAX_GLYPHS - left - 1
            text = f"{text[:left]}…{text[-right:]}"
        self.setText(f"[{text}]")


class BrowserView(QtWidgets.QFrame):
    commandRequested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("e3BrowserView")
        self.setMinimumWidth(255)
        self._scans = ()
        self._selected_frames: tuple[DisplayFrameKey, ...] = ()
        self._trace_frames: tuple[DisplayFrameKey, ...] = ()
        self._plot_mode = "Single"
        self._reconciling_frames = False
        self._has_committed_navigation = False
        self._committed_current: DisplayFrameKey | None = None
        self._committed_selected: tuple[DisplayFrameKey, ...] = ()
        self._pending_frame_command: ShellCommand | None = None
        self._frame_gesture_active = False
        self._frame_selection_explicit: bool | None = None
        self._frame_pointer_current: DisplayFrameKey | None = None
        layout = QtWidgets.QVBoxLayout(self)
        self._root_layout = layout
        layout.addLayout(self._make_menu_row())
        layout.addLayout(self._make_header())

        lists = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lists.setObjectName("e3BrowserLists")
        # A real gutter, not two bordered views touching edge-to-edge.  The
        # theme paints this handle as panel background; keeping a splitter
        # preserves the user's ability to give long scan names more room.
        lists.setChildrenCollapsible(False)
        self.list_splitter = lists

        scan_pane = QtWidgets.QWidget()
        scan_pane_layout = QtWidgets.QVBoxLayout(scan_pane)
        scan_pane_layout.setContentsMargins(0, 0, 0, 0)
        self._scan_pane_layout = scan_pane_layout
        scan_labels = QtWidgets.QWidget(scan_pane)
        scan_labels_layout = QtWidgets.QHBoxLayout(scan_labels)
        scan_labels_layout.setContentsMargins(0, 0, 0, 0)
        scan_labels_layout.setSpacing(4)
        self.scans_label = QtWidgets.QLabel("Scans")
        self.directory_label = _ElidedDirectoryLabel()
        scan_labels_layout.addWidget(self.scans_label)
        scan_labels_layout.addWidget(self.directory_label, 1)
        self.scans = QtWidgets.QListWidget()
        self.scans.setObjectName("e3ScanList")
        scan_pane_layout.addWidget(scan_labels)
        scan_pane_layout.addWidget(self.scans, 1)

        frame_pane = QtWidgets.QWidget()
        frame_pane_layout = QtWidgets.QVBoxLayout(frame_pane)
        frame_pane_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_pane_layout = frame_pane_layout
        self.frames_label = QtWidgets.QLabel("Frames")
        self.frames = QtWidgets.QListView()
        self.frames.setObjectName("e3BrowserFrameList")
        self.frame_model = FrameListModel(self.frames)
        self.frames.setModel(self.frame_model)
        self.frames.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._frame_click_filter = _AccumulatingFrameClickFilter(
            self.frames,
            lambda: self._plot_mode,
            self._set_frame_selection_intent,
        )
        self.frames.viewport().installEventFilter(
            self._frame_click_filter
        )
        self._frame_cadence_filter = _FrameSelectionCadenceFilter(self)
        self.frames.installEventFilter(self._frame_cadence_filter)
        self._frame_selection_coalescer = Coalescer(
            _FRAME_SELECTION_SETTLE_MS,
            mode="debounce",
            parent=self,
        )
        self._frame_selection_coalescer.triggered.connect(
            self._flush_frame_selection
        )
        frame_pane_layout.addWidget(self.frames_label)
        frame_pane_layout.addWidget(self.frames, 1)

        lists.addWidget(scan_pane)
        lists.addWidget(frame_pane)
        lists.setStretchFactor(0, 3)
        lists.setStretchFactor(1, 1)
        layout.addWidget(lists, 1)

        actions = QtWidgets.QHBoxLayout()
        self._actions_layout = actions
        self.show_all = QtWidgets.QPushButton("Show All")
        self.metadata = QtWidgets.QPushButton("Metadata")
        self.auto_last = QtWidgets.QPushButton("Auto Last")
        for button in (self.show_all, self.metadata, self.auto_last):
            button.setObjectName("e3BrowserCompactButton")
        self.auto_last.setCheckable(True)
        for button in (self.show_all, self.metadata, self.auto_last):
            actions.addWidget(button)
        layout.addLayout(actions)

        self.scans.itemSelectionChanged.connect(self._scan_selected)
        self.frames.selectionModel().selectionChanged.connect(
            self._frames_selected
        )
        self.show_all.clicked.connect(
            lambda: self._emit(ShellCommandKind.SHOW_ALL)
        )
        self.metadata.clicked.connect(
            lambda: self._emit(ShellCommandKind.SHOW_METADATA)
        )
        self.auto_last.toggled.connect(
            lambda checked: self._emit(
                ShellCommandKind.SET_AUTO_LAST, bool(checked)
            )
        )
        self._apply_spacing()

    @property
    def frame_selection_pending(self) -> bool:
        """Whether Qt has newer frame intent than the committed projection."""

        return (
            self._frame_gesture_active
            or self._pending_frame_command is not None
            or self._frame_selection_coalescer.is_pending()
        )

    def cancel_pending_frame_selection(self) -> None:
        """Drop unsettled UI selection intent at a terminal boundary."""

        self._cancel_pending_frame_selection()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if (
            event.type()
            in {QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.FontChange}
            and hasattr(self, "_root_layout")
        ):
            self._apply_spacing()

    def _apply_spacing(self) -> None:
        tokens = current_spacing_tokens()
        margin = tokens.panel_margin
        self._root_layout.setContentsMargins(
            margin,
            margin,
            margin,
            max(2, margin // 2),
        )
        self._root_layout.setSpacing(tokens.layout_gap)
        pane_gap = max(2, tokens.layout_gap // 2)
        self._scan_pane_layout.setSpacing(pane_gap)
        self._frame_pane_layout.setSpacing(pane_gap)
        self._actions_layout.setSpacing(tokens.layout_gap)
        self.list_splitter.setHandleWidth(tokens.browser_gap)

    def _make_menu_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        for title, object_name, actions in (
            (
                "File",
                "fileMenuButton",
                ("Open Folder", "New", "Save As", "Export"),
            ),
            (
                "Config",
                "configMenuButton",
                ("Save", "Load", "Advanced"),
            ),
            ("Help", "helpMenuButton", ("Help",)),
        ):
            button = QtWidgets.QToolButton()
            button.setText(title)
            button.setObjectName(object_name)
            button.setPopupMode(
                QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
            )
            menu = QtWidgets.QMenu(button)
            for label in actions:
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, value=f"{title}:{label}": self._emit(
                        ShellCommandKind.MENU, value
                    )
                )
            if title == "Config":
                self._heavy_residency_menu = QtWidgets.QMenu("Heavy residency", menu); menu.addMenu(self._heavy_residency_menu)
                self._heavy_residency_group = QtGui.QActionGroup(self._heavy_residency_menu)
                self._heavy_residency_group.setExclusive(True)
                self._heavy_residency_actions = {}
                for label in ("Auto", "16", "32", "64"):
                    action = self._heavy_residency_menu.addAction(label)
                    action.setCheckable(True)
                    self._heavy_residency_group.addAction(action)
                    action.triggered.connect(
                        lambda _checked=False, choice=label: self._emit(
                            ShellCommandKind.MENU,
                            f"Config:Heavy residency:{choice}",
                        )
                    )
                    self._heavy_residency_actions[label.lower()] = action
                self._heavy_residency_actions["auto"].setChecked(True)
            button.setMenu(menu)
            row.addWidget(button)
        row.addStretch(1)
        return row

    def reconcile_heavy_residency(
        self, choice: str, *, next_run: bool,
    ) -> None:
        action = self._heavy_residency_actions.get(str(choice).lower())
        if action is None:
            raise ValueError("unsupported heavy residency choice")
        action.setChecked(True)
        self._heavy_residency_menu.setTitle(
            "Heavy residency (next run)" if next_run else "Heavy residency"
        )

    def _make_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("DATA BROWSER")
        title.setObjectName("dataBrowserHeader")
        self.date_sort = QtWidgets.QPushButton("Time")
        self.date_sort.setObjectName("e3BrowserCompactButton")
        self.date_sort.setCheckable(True)
        self.refresh = QtWidgets.QToolButton()
        self.refresh.setObjectName("e3RefreshBrowser")
        self.refresh.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_BrowserReload
            )
        )
        self.refresh.setToolTip("Refresh")
        self.refresh.setAccessibleName("Refresh")
        self.refresh.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        row.addWidget(title, 1)
        row.addWidget(self.date_sort)
        row.addWidget(self.refresh)
        self.date_sort.toggled.connect(
            lambda checked: self._emit(
                ShellCommandKind.SET_DATE_SORT, bool(checked)
            )
        )
        self.refresh.clicked.connect(
            lambda: self._emit(ShellCommandKind.REFRESH_BROWSER)
        )
        return row

    def reconcile(
        self,
        state: BrowserProjection,
        navigation: FrameNavigationProjection,
        *,
        plot_mode: str,
    ) -> None:
        authoritative_changed = (
            self._has_committed_navigation
            and (
                navigation.current is not self._committed_current
                or not _same_frame_identities(
                    navigation.selected,
                    self._committed_selected,
                )
            )
        )
        pending = self._pending_frame_command
        if pending is not None and plot_mode != self._plot_mode:
            self._cancel_pending_frame_selection()
            pending = None
        if pending is not None and not _command_belongs_to_catalog(
            pending,
            state.frames,
        ):
            self._cancel_pending_frame_selection()
            pending = None
        if (
            pending is not None
            and authoritative_changed
            and plot_mode in _VISIT_ACCUMULATING_PLOT_MODES
        ):
            membership = (
                navigation.selected
                if navigation.selected
                else (() if pending.frame is None else (pending.frame,))
            )
            pending = ShellCommand(
                pending.kind,
                pending.value,
                pending.path,
                pending.frame,
                membership,
            )
            self._pending_frame_command = pending
            self._trace_frames = membership
        if authoritative_changed and pending is None:
            self._cancel_pending_frame_selection()
        self._committed_current = navigation.current
        self._committed_selected = navigation.selected
        self._has_committed_navigation = True

        blockers = [
            QtCore.QSignalBlocker(widget)
            for widget in (
                self,
                self.scans,
                self.date_sort,
                self.auto_last,
            )
        ]
        self.directory_label.set_path(state.directory)
        if state.scans != self._scans:
            self.scans.clear()
            for scan in state.scans:
                item = QtWidgets.QListWidgetItem(scan.label)
                item.setData(_USER_ROLE, scan.identifier)
                item.setToolTip(scan.detail)
                self.scans.addItem(item)
            self._scans = state.scans
        for index in range(self.scans.count()):
            item = self.scans.item(index)
            item.setSelected(item.data(_USER_ROLE) == state.selected_scan)

        self._plot_mode = plot_mode
        frames = state.frames
        changed = self.frame_model.reconcile(frames)
        if pending is None:
            current = (
                navigation.current
                if any(
                    candidate is navigation.current
                    for candidate in frames
                )
                else None
            )
            if plot_mode in _VISIT_ACCUMULATING_PLOT_MODES:
                trace_frames = tuple(
                    frame
                    for frame in navigation.selected
                    if any(candidate is frame for candidate in frames)
                )
                selected_frames = tuple(
                    frame
                    for frame in self._selected_frames
                    if any(candidate is frame for candidate in frames)
                )
                if (changed or not selected_frames) and trace_frames:
                    selected_frames = (
                        () if current is None else (current,)
                    )
            else:
                selected_frames = tuple(
                    frame
                    for frame in navigation.selected
                    if any(candidate is frame for candidate in frames)
                )
                trace_frames = selected_frames
        else:
            selected_frames = self._selected_frames
            current = pending.frame
            trace_frames = self._trace_frames

        selection = self.frames.selectionModel()
        self._reconciling_frames = True
        if changed or not _same_frame_identities(
            selected_frames,
            self._selected_frames,
        ) or not _same_frame_identities(
            selected_frames,
            self._ui_selected_frames(),
        ):
            selection.clearSelection()
            for frame in selected_frames:
                row = self.frame_model.row_for(frame)
                if row is not None:
                    selection.select(
                        self.frame_model.index(row, 0),
                        QtCore.QItemSelectionModel.SelectionFlag.Select
                        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )
            self._selected_frames = selected_frames
        self._trace_frames = trace_frames
        if current is None:
            selection.setCurrentIndex(
                QtCore.QModelIndex(),
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        else:
            row = self.frame_model.row_for(current)
            if row is not None:
                selection.setCurrentIndex(
                    self.frame_model.index(row, 0),
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
        self._reconciling_frames = False
        self.date_sort.setChecked(state.date_sorted)
        self.auto_last.setChecked(state.auto_last)
        del blockers

    def _scan_selected(self) -> None:
        items = self.scans.selectedItems()
        if items:
            self._cancel_pending_frame_selection()
            self._emit(
                ShellCommandKind.SELECT_SCAN,
                items[0].data(_USER_ROLE),
            )

    def _frames_selected(
        self,
        selected: QtCore.QItemSelection,
        deselected: QtCore.QItemSelection,
    ) -> None:
        if (
            self._reconciling_frames
            or (not selected.indexes() and not deselected.indexes())
        ):
            return
        ui_selected_frames = tuple(
            index.data(QtCore.Qt.ItemDataRole.UserRole)
            for index in sorted(
                self.frames.selectionModel().selectedRows(),
                key=lambda candidate: candidate.row(),
            )
        )
        if not all(
            type(item) is DisplayFrameKey for item in ui_selected_frames
        ):
            return
        ui_selected_frames = tuple(ui_selected_frames)
        newly_selected = tuple(
            index.data(QtCore.Qt.ItemDataRole.UserRole)
            for index in selected.indexes()
            if index.column() == 0
        )
        indexed_current = self.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        current = next(
            (
                frame
                for frame in reversed(newly_selected)
                if type(frame) is DisplayFrameKey
            ),
            indexed_current
            if type(indexed_current) is DisplayFrameKey
            else (
                ui_selected_frames[-1]
                if ui_selected_frames
                else self._committed_current
            ),
        )
        if type(self._frame_pointer_current) is DisplayFrameKey:
            current = self._frame_pointer_current

        selected_frames = ui_selected_frames
        if self._plot_mode in _VISIT_ACCUMULATING_PLOT_MODES:
            selected_frames = (
                ui_selected_frames
                if self._frame_selection_explicit is True
                else _catalog_ordered_membership(
                    self.frame_model.frames,
                    self._trace_frames,
                    () if current is None else (current,),
                )
            )

        self._pending_frame_command = ShellCommand(
            ShellCommandKind.SELECT_BROWSER_FRAMES,
            frame=current,
            frames=selected_frames,
        )
        self._selected_frames = ui_selected_frames
        self._trace_frames = selected_frames
        if not self._frame_gesture_active:
            self._frame_selection_explicit = None
            self._frame_selection_coalescer.trigger()

    def _set_frame_selection_intent(
        self,
        explicit: bool | None,
        current: DisplayFrameKey | None,
    ) -> None:
        if (
            explicit is False
            and current is not self.frames.currentIndex().data(_USER_ROLE)
            and _same_frame_identities(self._ui_selected_frames(), (current,))
        ):
            self._trace_frames = _catalog_ordered_membership(
                self.frame_model.frames, self._trace_frames, (current,)
            )
            self._pending_frame_command = ShellCommand(
                ShellCommandKind.SELECT_BROWSER_FRAMES,
                frame=current,
                frames=self._trace_frames,
            )
            self._frame_selection_coalescer.trigger()
        self._frame_selection_explicit = explicit
        self._frame_pointer_current = current

    def _begin_frame_gesture(self, modifiers) -> None:
        if self._frame_gesture_active:
            return
        self._frame_selection_coalescer.cancel()
        self._frame_selection_explicit = bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ShiftModifier
                | QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
        )
        self._frame_gesture_active = True

    def _finish_frame_gesture(self) -> None:
        if not self._frame_gesture_active:
            return
        self._frame_gesture_active = False
        if self._pending_frame_command is not None:
            self._frame_selection_coalescer.trigger()
        self._frame_selection_explicit = None

    def _ui_selected_frames(self) -> tuple[DisplayFrameKey, ...]:
        frames = tuple(
            index.data(QtCore.Qt.ItemDataRole.UserRole)
            for index in sorted(
                self.frames.selectionModel().selectedRows(),
                key=lambda candidate: candidate.row(),
            )
        )
        return (
            tuple(frames)
            if all(type(frame) is DisplayFrameKey for frame in frames)
            else ()
        )

    def _flush_frame_selection(self) -> None:
        command = self._pending_frame_command
        self._pending_frame_command = None
        if command is not None:
            self.commandRequested.emit(command)

    def _cancel_pending_frame_selection(self) -> None:
        self._frame_selection_coalescer.cancel()
        self._pending_frame_command = None
        self._frame_gesture_active = False
        self._frame_selection_explicit = None

    def _emit(self, kind: ShellCommandKind, value=None) -> None:
        self.commandRequested.emit(ShellCommand(kind, value))


def _same_frame_identities(
    left: tuple[DisplayFrameKey, ...],
    right: tuple[DisplayFrameKey, ...],
) -> bool:
    return len(left) == len(right) and all(
        first is second for first, second in zip(left, right)
    )


def _catalog_ordered_membership(catalog, *groups):
    return tuple(
        frame
        for frame in catalog
        if any(frame is member for group in groups for member in group)
    )


def _command_belongs_to_catalog(
    command: ShellCommand,
    catalog: tuple[DisplayFrameKey, ...],
) -> bool:
    return (
        (
            command.frame is None
            or any(frame is command.frame for frame in catalog)
        )
        and all(
            any(frame is candidate for candidate in catalog)
            for frame in command.frames
        )
    )


__all__ = ["BrowserView"]
