"""Passive Data Browser rendering for the E3 visual shell."""

from __future__ import annotations

import os
from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.themes.spacing import current_spacing_tokens
from xdart.utils.throttle import Coalescer

from .browser_model import FrameListModel
from .display_values import DisplayFrameKey
from .shell_values import (
    BrowserProjection,
    FrameSelectionIntent,
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)


_USER_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_DIRECTORY_ROLE = _USER_ROLE + 1
_MUTED_ROLE = _USER_ROLE + 2
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
        if os.environ.get("XDART_VIEWER_DEBUG") == "1":
            focus = QtWidgets.QApplication.focusWidget()
            print("viewer_frame_click", {"mode": plot_mode, "row": index.row(),
                "focus": None if focus is None else focus.objectName()}, flush=True)
        selection = self._view.selectionModel()
        anchor = selection.currentIndex()
        intent = (
            FrameSelectionIntent.REMOVE_TRACE_RANGE
            if visit_mode and shift_modifier
            else FrameSelectionIntent.TOGGLE_TRACE
            if visit_mode and toggle_modifier
            else FrameSelectionIntent.VISIT
            if visit_mode
            else FrameSelectionIntent.EXACT
        )
        self._set_selection_intent(
            intent,
            index.data(_USER_ROLE),
            anchor.data(_USER_ROLE) if anchor.isValid() else None,
        )
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
                if (toggle_modifier or not visit_mode) and not shift_modifier
                else QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        self._set_selection_intent(None, None, None)
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
                if event.matches(QtGui.QKeySequence.StandardKey.SelectAll):
                    owner._select_all_frames()
                    return True
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


class _ArtifactNavigationFilter(QtCore.QObject):
    """Plain arrows visit adjacent files; directory entry is explicit."""

    def __init__(self, view, activate_directory, visit_artifact):
        super().__init__(view)
        self._view = view
        self._activate_directory = activate_directory
        self._visit_artifact = visit_artifact

    def eventFilter(self, watched, event) -> bool:
        event_type = event.type()
        if event_type not in {QtCore.QEvent.Type.MouseButtonPress,
                              QtCore.QEvent.Type.MouseButtonDblClick,
                              QtCore.QEvent.Type.KeyPress}:
            return False
        view = self._view
        if event_type == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                return False
            index = view.indexAt(event.position().toPoint())
            if not index.isValid():
                return False
            modifiers = event.modifiers()
            if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
                intent = FrameSelectionIntent.REMOVE_TRACE_RANGE
            elif modifiers & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            ):
                intent = FrameSelectionIntent.TOGGLE_TRACE
            elif modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
                intent = FrameSelectionIntent.VISIT
            else:
                return False
            if self._visit_artifact(view.item(index.row()), intent):
                return True
            return False
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            index = view.indexAt(event.position().toPoint())
            item = view.item(index.row()) if index.isValid() else None
            if item is not None and item.data(_DIRECTORY_ROLE):
                view.setCurrentRow(index.row())
                self._activate_directory(item)
                return True
            return False
        if (event.type() == QtCore.QEvent.Type.KeyPress
                and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)):
            item = view.currentItem()
            if item is not None and item.data(_DIRECTORY_ROLE):
                self._activate_directory(item)
                return True
            return False
        if (event.type() != QtCore.QEvent.Type.KeyPress
                or event.modifiers() not in {
                    QtCore.Qt.KeyboardModifier.NoModifier,
                    QtCore.Qt.KeyboardModifier.KeypadModifier,
                }
                or event.key() not in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down)):
            return False
        step = -1 if event.key() == QtCore.Qt.Key.Key_Up else 1
        row = view.currentRow()
        if row < 0:
            row = view.count() if step < 0 else -1
        for index in range(row + step, view.count() if step > 0 else -1, step):
            if not view.item(index).data(_DIRECTORY_ROLE):
                if not self._visit_artifact(view.item(index)):
                    view.setCurrentRow(index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
                break
        return True


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


#: Int's Data Browser menus: (title, its actions); each button's object name is ``<title>MenuButton``.
INT_BROWSER_MENUS = (
    ("File", ("Open Folder", "New", "Save As", "Export")),
    ("Config", ("Save", "Load", "Advanced", "Performance Diagnostics…")),
    ("Help", ("Help", "Export Analyze Results Notebook")),
)


@dataclass(frozen=True, slots=True)
class BrowserProfile:
    """What a page's Data Browser shows (AUD-9; the maintainer's reuse rule: another page mounts this view
    instead of copying its menus, heading and lists). The default is Int's. A results profile (``results``)
    lists a page's own results, filled through :meth:`BrowserView.reconcile_results`; choosing one emits
    ``SELECT_RESULT`` and Open… emits ``OPEN_RESULT``."""

    menus: tuple[tuple[str, tuple[str, ...]], ...] = INT_BROWSER_MENUS
    #: Int's Config submenus (raw image resolution, heavy residency)
    raw_resolution_menus: bool = True
    list_caption: str = "Scans"
    directory: bool = True
    frames: bool = True
    frames_caption: str = "Frames"
    #: which actions show, of "Show All", "Metadata", "Auto Last", "Open…" and "Raw image"
    actions: tuple[str, ...] = ("Show All", "Metadata", "Auto Last")
    date_sort: bool = True
    refresh: bool = True
    results: bool = False
    #: a results list's selection: "single" (one result, ``SELECT_RESULT``) or "extended" (several, for an
    #: overlay: ``SELECT_RESULT`` when the current one changes and ``SELECT_RESULTS`` when the selection does)
    result_selection: str = "single"
    #: a results list's second pane: the current result's own entries (Stitch's contributing frames), filled
    #: through :meth:`BrowserView.reconcile_result_frames`; choosing one emits ``SELECT_RESULT_FRAME``
    result_frames: bool = False
    #: the "DATA BROWSER" heading row (a keyed list inside a tool window has none: ``keyed_list_profile``)
    heading: bool = True


def keyed_list_profile(caption: str) -> BrowserProfile:
    """A plain keyed list (FIT-TOOLS-2026-10, Codex's design review F7: the Phase Fit's CIF rows): a window's own
    entries under ``caption``, each keyed by its identity (``BrowserResult``), with no menus, heading, frames or
    actions and none of a scan's or a frame's meaning. The window keeps its collection and reconciles the list from
    it after each Add or Remove (``reconcile_results``); ``current_result`` is the entry chosen."""
    return BrowserProfile(menus=(), raw_resolution_menus=False, list_caption=caption, directory=False, frames=False,
                          actions=(), date_sort=False, refresh=False, results=True, heading=False)


@dataclass(frozen=True, slots=True)
class BrowserResult:
    """One entry of a results profile's list: its identity, the text shown and the tooltip; ``muted`` shows it
    in the disabled text colour (a contributing frame the result excluded)."""

    identifier: str
    label: str
    detail: str = ""
    muted: bool = False


class BrowserView(QtWidgets.QFrame):
    commandRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        profile: BrowserProfile | None = None,
    ) -> None:
        super().__init__(parent)
        self.profile = BrowserProfile() if profile is None else profile
        if type(self.profile) is not BrowserProfile:
            raise TypeError("profile must be an exact BrowserProfile")
        if self.profile.result_selection not in {"single", "extended"}:
            raise ValueError("result_selection must be 'single' or 'extended'")
        if self.profile.result_frames and not self.profile.results:
            raise ValueError("result_frames needs a results profile")
        self._results: tuple[BrowserResult, ...] = ()
        self.setObjectName("e3BrowserView")
        self.setMinimumWidth(255)
        self._scans = ()
        self._current_artifact = ""
        self._artifact_selection_contract = None
        self._applied_artifact_selection: tuple[str, ...] = ()
        self._applied_current_artifact: str | None = None
        self._selected_frames: tuple[DisplayFrameKey, ...] = ()
        self._trace_frames: tuple[DisplayFrameKey, ...] = ()
        self._plot_mode = "Single"
        self._reconciling_frames = False
        self._has_committed_navigation = False
        self._committed_current: DisplayFrameKey | None = None
        self._committed_selected: tuple[DisplayFrameKey, ...] = ()
        self._pending_frame_command: ShellCommand | None = None
        self._frame_gesture_active = False
        self._frame_selection_intent: FrameSelectionIntent | None = None
        self._frame_pointer_current: DisplayFrameKey | None = None
        self._frame_range_anchor: DisplayFrameKey | None = None
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
        self.scans_label = QtWidgets.QLabel(self.profile.list_caption)
        self.directory_label = _ElidedDirectoryLabel()
        if not self.profile.directory:
            self.directory_label.hide()
        scan_labels_layout.addWidget(self.scans_label)
        scan_labels_layout.addWidget(self.directory_label, 1)
        self.scans = QtWidgets.QListWidget()
        self.scans.setObjectName("e3ScanList")
        self._artifact_navigation_filter = _ArtifactNavigationFilter(
            self.scans, self._activate_directory, self._visit_artifact,
        )
        if not self.profile.results:  # a results list has no directories or artifacts to visit
            self.scans.installEventFilter(self._artifact_navigation_filter)
            self.scans.viewport().installEventFilter(self._artifact_navigation_filter)
        scan_pane_layout.addWidget(scan_labels)
        scan_pane_layout.addWidget(self.scans, 1)

        frame_pane = QtWidgets.QWidget()
        frame_pane_layout = QtWidgets.QVBoxLayout(frame_pane)
        frame_pane_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_pane_layout = frame_pane_layout
        self.frames_label = QtWidgets.QLabel(self.profile.frames_caption)
        self.frames = QtWidgets.QListView()
        self.frames.setUniformItemSizes(True)
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
        #: a results profile's second list: the current result's entries (``reconcile_result_frames``)
        self.result_frames = None
        if self.profile.result_frames:
            self.result_frames = QtWidgets.QListWidget()
            self.result_frames.setObjectName("e3BrowserResultFrames")
            self.frames.hide()  # Int's frame list keeps Int's frame identities; this pane lists the result's
            frame_pane_layout.addWidget(self.result_frames, 1)

        lists.addWidget(scan_pane)
        lists.addWidget(frame_pane)
        if not self.profile.frames:
            frame_pane.hide()
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
        self._action_buttons = {"Show All": self.show_all, "Metadata": self.metadata,
                                "Auto Last": self.auto_last}
        for label, button in tuple(self._action_buttons.items()):
            actions.addWidget(button)
            if label not in self.profile.actions:
                button.hide()
        self.open_result = None
        if "Open…" in self.profile.actions:
            self.open_result = QtWidgets.QPushButton("Open…")
            self.open_result.setObjectName("e3BrowserCompactButton")
            self._action_buttons["Open…"] = self.open_result
            actions.addWidget(self.open_result)
            self.open_result.clicked.connect(
                lambda: self._emit(ShellCommandKind.OPEN_RESULT)
            )
        self.raw_image = None
        if "Raw image" in self.profile.actions:
            self.raw_image = QtWidgets.QPushButton("Raw image")
            self.raw_image.setObjectName("e3BrowserCompactButton")
            self._action_buttons["Raw image"] = self.raw_image
            actions.addWidget(self.raw_image)
            self.raw_image.clicked.connect(
                lambda: self._emit(ShellCommandKind.SHOW_RAW_IMAGE)
            )
        layout.addLayout(actions)

        if self.profile.results and self.profile.result_selection == "extended":
            self.scans.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )
            # as the list's own signals: the current result, then the selection, each as Qt reports it
            self.scans.currentItemChanged.connect(
                lambda current, _previous: self._emit(
                    ShellCommandKind.SELECT_RESULT,
                    None if current is None else current.data(_USER_ROLE),
                )
            )
            self.scans.itemSelectionChanged.connect(
                lambda: self.commandRequested.emit(ShellCommand(
                    ShellCommandKind.SELECT_RESULTS, artifacts=self.selected_results()))
            )
        elif self.profile.results:
            self.scans.currentItemChanged.connect(self._result_selected)
        else:
            self.scans.itemSelectionChanged.connect(self._scan_selected)
            self.scans.itemClicked.connect(self._activate_directory)
            self.scans.itemActivated.connect(self._activate_directory)
        self.frames.selectionModel().selectionChanged.connect(
            self._frames_selected
        )
        if self.result_frames is not None:
            self.result_frames.currentItemChanged.connect(
                lambda current, _previous: self._emit(
                    ShellCommandKind.SELECT_RESULT_FRAME,
                    None if current is None else current.data(_USER_ROLE),
                )
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
        #: each menu by its title: a page hands its Config and Help to the host (``AppMenuHosts``)
        self.menus: dict[str, QtWidgets.QMenu] = {}
        for title, actions in self.profile.menus:
            object_name = f"{title.lower()}MenuButton"
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
            self.menus[title] = menu
            if title == "Config" and self.profile.raw_resolution_menus:
                self._detector_menu = QtWidgets.QMenu("Raw image resolution", menu)
                menu.addMenu(self._detector_menu)
                self._detector_group = QtGui.QActionGroup(self._detector_menu)
                self._detector_group.setExclusive(True)
                self._detector_actions = {}
                for label, choice in (("Thumbnail", "thumbnail"), ("Full Raw", "full")):
                    action = self._detector_menu.addAction(label)
                    action.setCheckable(True)
                    self._detector_group.addAction(action)
                    action.triggered.connect(
                        lambda _checked=False, value=choice: self._emit(
                            ShellCommandKind.SET_DETECTOR_MODE, value))
                    self._detector_actions[choice] = action
                self._detector_actions["thumbnail"].setChecked(True)
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

    def reconcile_detector_mode(self, state) -> None:
        self._detector_actions[state.detector_mode].setChecked(True)
        full = self._detector_actions["full"]
        full.setEnabled(state.detector_available)
        full.setToolTip(state.detector_diagnostic if not state.detector_available else "")

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
        self.heading = title
        if not self.profile.heading:
            title.hide()
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
        if not self.profile.date_sort:
            self.date_sort.hide()
        if not self.profile.refresh:
            self.refresh.hide()
        self.date_sort.toggled.connect(
            lambda checked: self._emit(
                ShellCommandKind.SET_DATE_SORT, bool(checked)
            )
        )
        self.refresh.clicked.connect(
            lambda: self._emit(ShellCommandKind.REFRESH_BROWSER)
        )
        return row

    def reconcile_results(
        self, entries: tuple[BrowserResult, ...], current: str | None,
    ) -> None:
        """A results profile's list: its entries in order, and the one current (None: none). The list is
        rebuilt only when the entries change; neither a rebuild nor moving the current entry emits."""
        if not self.profile.results:
            raise RuntimeError("reconcile_results needs a results profile")
        entries = tuple(entries)
        if any(type(entry) is not BrowserResult for entry in entries):
            raise TypeError("results must be exact BrowserResult entries")
        blocker = QtCore.QSignalBlocker(self.scans)
        # against the list itself, not what was last published: a list cleared from outside is filled again
        if _listed_entries(self.scans) != entries:
            _fill(self.scans, entries)
        self._results = entries
        target = next((self.scans.item(index) for index in range(self.scans.count())
                       if self.scans.item(index).data(_USER_ROLE) == current), None)
        if self.scans.currentItem() is not target:
            self.scans.setCurrentItem(target)
        del blocker

    def _result_selected(self, current, _previous) -> None:
        if current is not None:
            self._emit(ShellCommandKind.SELECT_RESULT, current.data(_USER_ROLE))

    def current_result(self) -> str | None:
        """A results list's current entry's identity (None: none)."""
        item = self.scans.currentItem()
        return None if item is None else item.data(_USER_ROLE)

    def current_result_row(self) -> int:
        """A results list's current row (-1: none)."""
        return self.scans.currentRow()

    def current_result_label(self) -> str:
        """The text the list shows for its current entry ("" for none)."""
        item = self.scans.currentItem()
        return "" if item is None else item.text()

    def selected_results(self) -> tuple[str, ...]:
        """A results list's selected entries' identities, in the order Qt reports them (an overlay's order)."""
        return tuple(item.data(_USER_ROLE) for item in self.scans.selectedItems())

    def set_current_result(self, identifier: str | None, *, keep_selection: bool) -> None:
        """Make an entry current as a user would (its commands are emitted): ``keep_selection`` keeps the other
        selected entries (an overlay stepping through them), else the selection becomes this entry alone.
        None, or an entry not listed: no current entry."""
        row = next((index for index in range(self.scans.count())
                    if self.scans.item(index).data(_USER_ROLE) == identifier), -1)
        flag = (QtCore.QItemSelectionModel.SelectionFlag.NoUpdate if keep_selection
                else QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.scans.setCurrentRow(row, flag)

    def clear_result_selection(self) -> None:
        """No entry selected or current, as a user clearing the list would (its commands are emitted)."""
        self.scans.clearSelection()
        self.scans.setCurrentRow(-1)

    def reconcile_result_frames(
        self, entries: tuple[BrowserResult, ...], current: str | None,
    ) -> None:
        """The second pane's entries (a results profile with ``result_frames``), rebuilt on every call: the
        page calls it when it shows a result. Neither the rebuild nor the current entry emits."""
        if self.result_frames is None:
            raise RuntimeError("reconcile_result_frames needs a result_frames profile")
        entries = tuple(entries)
        if any(type(entry) is not BrowserResult for entry in entries):
            raise TypeError("result frames must be exact BrowserResult entries")
        blocker = QtCore.QSignalBlocker(self.result_frames)
        _fill(self.result_frames, entries)
        target = next((self.result_frames.item(index) for index in range(self.result_frames.count())
                       if self.result_frames.item(index).data(_USER_ROLE) == current), None)
        self.result_frames.setCurrentItem(target)
        del blocker

    def current_result_frame(self) -> str | None:
        """The second pane's current entry's identity (None: none)."""
        item = None if self.result_frames is None else self.result_frames.currentItem()
        return None if item is None else item.data(_USER_ROLE)

    def current_result_frame_label(self) -> str:
        """The text the second pane shows for its current entry ("" for none)."""
        item = None if self.result_frames is None else self.result_frames.currentItem()
        return "" if item is None else item.text()

    def result_frame_count(self) -> int:
        return 0 if self.result_frames is None else self.result_frames.count()

    def set_action_enabled(self, action: str, enabled: bool) -> None:
        """Enable or disable one of the profile's actions ("Show All", "Metadata", "Auto Last", "Open…",
        "Raw image")."""
        self._action_buttons[action].setEnabled(bool(enabled))

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
            pending, state.frames
        ):
            self._cancel_pending_frame_selection()
            pending = None
        if (
            pending is not None
            and authoritative_changed
            and plot_mode in _VISIT_ACCUMULATING_PLOT_MODES
        ):
            self._trace_frames = _membership_after_intent(
                state.frames,
                navigation.selected,
                pending.frames,
                pending.intent,
            )
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
        selection_mode = (
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            if state.multi_artifact_selection
            else QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        selection_mode_changed = self.scans.selectionMode() != selection_mode
        if selection_mode_changed:
            self.scans.setSelectionMode(selection_mode)
        scans_changed = state.scans != self._scans
        if scans_changed:
            self.scans.clear()
            for scan in state.scans:
                item = QtWidgets.QListWidgetItem(scan.label)
                item.setData(_USER_ROLE, scan.identifier)
                item.setData(_DIRECTORY_ROLE, scan.is_directory)
                item.setToolTip(scan.detail)
                self.scans.addItem(item)
            self._scans = state.scans
        selected_artifacts = (
            state.selected_artifacts
            or (() if not state.selected_scan else (state.selected_scan,))
        )
        highlighted_artifacts = (
            (state.selected_scan,)
            if (
                plot_mode in _VISIT_ACCUMULATING_PLOT_MODES
                and state.selected_scan
            )
            else selected_artifacts
        )
        selection_contract = (
            selected_artifacts, state.selected_scan, highlighted_artifacts,
        )
        current_item = self.scans.currentItem()
        current_identifier = (
            None if current_item is None else current_item.data(_USER_ROLE)
        )
        selected_identifiers = tuple(
            item.data(_USER_ROLE) for item in self.scans.selectedItems()
        )
        reconcile_artifact_selection = (
            scans_changed
            or selection_mode_changed
            or selection_contract != self._artifact_selection_contract
            or selected_identifiers != self._applied_artifact_selection
            or current_identifier != self._applied_current_artifact
        )
        if os.environ.get("XDART_VIEWER_DEBUG") == "1" and reconcile_artifact_selection:
            print("viewer_artifact_projection", {
                "mode": plot_mode, "ui_current": current_identifier,
                "projected_current": state.selected_scan,
                "projected_paths": selected_artifacts,
            }, flush=True)
        if reconcile_artifact_selection:
            current_artifact_item = None
            for index in range(self.scans.count()):
                item = self.scans.item(index)
                item.setSelected(item.data(_USER_ROLE) in highlighted_artifacts)
                if item.data(_USER_ROLE) == state.selected_scan:
                    current_artifact_item = item
            if current_artifact_item is not None:
                self.scans.setCurrentItem(
                    current_artifact_item,
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
            else:
                self.scans.setCurrentItem(
                    None,
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
            self._artifact_selection_contract = selection_contract
            self._applied_artifact_selection = tuple(
                item.data(_USER_ROLE) for item in self.scans.selectedItems()
            )
            current_item = self.scans.currentItem()
            self._applied_current_artifact = (
                None
                if current_item is None
                else current_item.data(_USER_ROLE)
            )
        self._current_artifact = state.selected_scan

        self._plot_mode = plot_mode
        frames = state.frames
        changed = self.frame_model.reconcile(frames)
        owns_frame = self.frame_model.owns
        if pending is None:
            current = (
                navigation.current
                if owns_frame(navigation.current)
                else None
            )
            if plot_mode in _VISIT_ACCUMULATING_PLOT_MODES:
                trace_frames = tuple(
                    frame
                    for frame in navigation.selected
                    if owns_frame(frame)
                )
                selected_frames = tuple(
                    frame
                    for frame in self._selected_frames
                    if owns_frame(frame)
                )
                if (changed or not selected_frames) and trace_frames:
                    selected_frames = (
                        () if current is None else (current,)
                    )
            else:
                selected_frames = tuple(
                    frame
                    for frame in navigation.selected
                    if owns_frame(frame)
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

    def _visit_artifact(
        self,
        item,
        intent: FrameSelectionIntent = FrameSelectionIntent.VISIT,
    ) -> bool:
        if (item.data(_DIRECTORY_ROLE)
                or self._plot_mode not in _VISIT_ACCUMULATING_PLOT_MODES
                or self.scans.selectionMode()
                != QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection):
            return False
        anchor = self.scans.currentItem()
        artifacts = (item.data(_USER_ROLE),)
        if intent is FrameSelectionIntent.REMOVE_TRACE_RANGE and anchor is not None:
            first, last = sorted((self.scans.row(anchor), self.scans.row(item)))
            artifacts = tuple(
                candidate.data(_USER_ROLE)
                for row in range(first, last + 1)
                if not (candidate := self.scans.item(row)).data(_DIRECTORY_ROLE)
            )
        blocker = QtCore.QSignalBlocker(self.scans)
        self.scans.setCurrentItem(
            item, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        del blocker
        if intent is FrameSelectionIntent.VISIT:
            self._scan_selected(intent)
        else:
            self.commandRequested.emit(ShellCommand(
                ShellCommandKind.SELECT_SCAN,
                item.data(_USER_ROLE),
                path=("artifact",),
                artifacts=artifacts,
                intent=intent,
            ))
        return True

    def _scan_selected(self, intent=FrameSelectionIntent.EXACT) -> None:
        items = tuple(
            self.scans.item(index)
            for index in range(self.scans.count())
            if self.scans.item(index).isSelected()
        )
        if not items:
            retained = next(
                (
                    self.scans.item(index)
                    for index in range(self.scans.count())
                    if self.scans.item(index).data(_USER_ROLE)
                    == self._current_artifact
                ),
                None,
            )
            if retained is not None:
                blocker = QtCore.QSignalBlocker(self.scans)
                retained.setSelected(True)
                self.scans.setCurrentItem(
                    retained,
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
                del blocker
            return
        self._cancel_pending_frame_selection()
        directories = tuple(
            item for item in items if item.data(_DIRECTORY_ROLE)
        )
        if directories:
            return
        artifacts = tuple(
            item.data(_USER_ROLE)
            for item in items
            if type(item.data(_USER_ROLE)) is str
            and item.data(_USER_ROLE)
        )
        if not artifacts:
            return
        current_item = self.scans.currentItem()
        current_row = self.scans.currentRow()
        current_artifact = (
            current_item.data(_USER_ROLE)
            if current_item in items
            and type(current_item.data(_USER_ROLE)) is str
            and current_item.data(_USER_ROLE)
            else self._current_artifact
            if self._current_artifact in artifacts
            else min(
                items,
                key=lambda item: (
                    abs(self.scans.row(item) - current_row),
                    self.scans.row(item),
                ),
            ).data(_USER_ROLE)
        )
        if os.environ.get("XDART_VIEWER_DEBUG") == "1":
            print("viewer_artifact_visit", {
                "intent": intent.value, "current": current_artifact,
                "paths": artifacts, "row": current_row,
            }, flush=True)
        self.commandRequested.emit(ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            current_artifact,
            path=("artifact",),
            artifacts=artifacts,
            intent=intent,
        ))

    def _select_all_frames(self) -> None:
        """Select All is exact membership, not one Overlay visit."""
        self._cancel_pending_frame_selection()
        frames = self.frame_model.frames
        if not frames:
            return
        current = self.frames.currentIndex().data(_USER_ROLE)
        if not any(current is frame for frame in frames):
            current = frames[-1]
        self._reconciling_frames = True
        try:
            self.frames.selectAll()
        finally:
            self._reconciling_frames = False
        self._selected_frames = self._trace_frames = frames
        self._queue_frame_command(ShellCommand(
            ShellCommandKind.SELECT_BROWSER_FRAMES,
            frame=current, frames=frames, intent=FrameSelectionIntent.EXACT,
        ))
        self._frame_selection_coalescer.trigger()

    def _activate_directory(self, item) -> None:
        if item.data(_DIRECTORY_ROLE) and len(self.scans.selectedItems()) == 1:
            # Directory reconciliation replaces this native item. Capture only
            # its value and let the input event finish before changing models.
            directory = item.data(_USER_ROLE)
            QtCore.QTimer.singleShot(
                0, self, lambda: self._emit(
                    ShellCommandKind.SELECT_SCAN, directory, path=("directory",),
                ),
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

        intent = FrameSelectionIntent.EXACT
        operands = ui_selected_frames
        if self._plot_mode in _VISIT_ACCUMULATING_PLOT_MODES:
            intent = self._frame_selection_intent or FrameSelectionIntent.VISIT
            if intent is FrameSelectionIntent.REMOVE_TRACE_RANGE:
                operands = _catalog_range(
                    self.frame_model.frames,
                    self._frame_range_anchor,
                    current,
                )
            else:
                operands = () if current is None else (current,)
            self._trace_frames = _membership_after_intent(
                self.frame_model.frames,
                self._trace_frames,
                operands,
                intent,
            )
        else:
            self._trace_frames = operands

        self._queue_frame_command(
            ShellCommand(
                ShellCommandKind.SELECT_BROWSER_FRAMES,
                frame=current,
                frames=operands,
                intent=intent,
            )
        )
        self._selected_frames = ui_selected_frames
        if not self._frame_gesture_active:
            self._frame_selection_intent = None
            self._frame_range_anchor = None
            self._frame_selection_coalescer.trigger()

    def _set_frame_selection_intent(
        self,
        intent: FrameSelectionIntent | None,
        current: DisplayFrameKey | None,
        anchor: DisplayFrameKey | None,
    ) -> None:
        if (
            intent is not None
            and not self._frame_gesture_active
            and self._pending_frame_command is not None
        ):
            self._frame_selection_coalescer.cancel()
            self._flush_frame_selection()
        if (
            intent is FrameSelectionIntent.VISIT
            and current is not self.frames.currentIndex().data(_USER_ROLE)
            and _same_frame_identities(self._ui_selected_frames(), (current,))
        ):
            self._trace_frames = _catalog_ordered_membership(
                self.frame_model.frames, self._trace_frames, (current,)
            )
            self._queue_frame_command(
                ShellCommand(
                    ShellCommandKind.SELECT_BROWSER_FRAMES,
                    frame=current,
                    frames=(current,),
                    intent=intent,
                )
            )
            self._frame_selection_coalescer.trigger()
        self._frame_selection_intent = intent
        self._frame_pointer_current = current
        self._frame_range_anchor = anchor

    def _begin_frame_gesture(self, modifiers) -> None:
        if self._frame_gesture_active:
            return
        self._frame_selection_coalescer.cancel()
        if self._pending_frame_command is not None:
            self._flush_frame_selection()
        visit_mode = self._plot_mode in _VISIT_ACCUMULATING_PLOT_MODES
        self._frame_selection_intent = (
            FrameSelectionIntent.REMOVE_TRACE_RANGE
            if visit_mode
            and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
            else FrameSelectionIntent.TOGGLE_TRACE
            if visit_mode
            and modifiers
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
            else FrameSelectionIntent.VISIT
            if visit_mode
            else FrameSelectionIntent.EXACT
        )
        anchor = self.frames.currentIndex().data(_USER_ROLE)
        self._frame_range_anchor = (
            anchor if type(anchor) is DisplayFrameKey else None
        )
        self._frame_gesture_active = True

    def _finish_frame_gesture(self) -> None:
        if not self._frame_gesture_active:
            return
        self._frame_gesture_active = False
        if self._pending_frame_command is not None:
            self._frame_selection_coalescer.trigger()
        self._frame_selection_intent = None
        self._frame_range_anchor = None

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
            if os.environ.get("XDART_VIEWER_DEBUG") == "1":
                print("viewer_selection", {
                    "intent": command.intent.value,
                    "current": getattr(command.frame, "local_frame_label", None),
                    "operands": [frame.local_frame_label for frame in command.frames],
                    "ui_current": self.frames.currentIndex().row(),
                    "gesture": self._frame_gesture_active,
                }, flush=True)
            self.commandRequested.emit(command)

    def _queue_frame_command(self, command: ShellCommand) -> None:
        pending = self._pending_frame_command
        if (
            pending is not None
            and self._frame_gesture_active
            and command.intent in {
                FrameSelectionIntent.VISIT,
                FrameSelectionIntent.TOGGLE_TRACE,
            }
        ):
            command = ShellCommand(
                command.kind,
                frame=command.frame,
                frames=_membership_after_intent(
                    self.frame_model.frames,
                    pending.frames,
                    command.frames,
                    command.intent,
                ),
                intent=command.intent,
            )
        self._pending_frame_command = command

    def _cancel_pending_frame_selection(self) -> None:
        self._frame_selection_coalescer.cancel()
        self._pending_frame_command = None
        self._frame_gesture_active = False
        self._frame_selection_intent = None
        self._frame_pointer_current = None
        self._frame_range_anchor = None

    def _emit(
        self, kind: ShellCommandKind, value=None, *, path: tuple[str, ...] = (),
    ) -> None:
        self.commandRequested.emit(ShellCommand(kind, value, path=path))


def _listed_entries(widget: QtWidgets.QListWidget) -> tuple[BrowserResult, ...]:
    """What a results list shows, as entries (its identities, texts, tooltips and muted marks)."""
    return tuple(
        BrowserResult(item.data(_USER_ROLE), item.text(), item.toolTip(),
                      item.data(_MUTED_ROLE) is True)
        for item in (widget.item(index) for index in range(widget.count()))
    )


def _fill(widget: QtWidgets.QListWidget, entries: tuple[BrowserResult, ...]) -> None:
    """A results list's entries, from scratch (one ``clear``)."""
    widget.clear()
    muted = widget.palette().color(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text)
    for entry in entries:
        item = QtWidgets.QListWidgetItem(entry.label)
        item.setData(_USER_ROLE, entry.identifier)
        item.setToolTip(entry.detail)
        if entry.muted:
            item.setData(_MUTED_ROLE, True)
            item.setForeground(muted)
        widget.addItem(item)


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


def _catalog_range(catalog, anchor, endpoint):
    positions = {
        id(frame): index for index, frame in enumerate(catalog)
    }
    if id(anchor) not in positions or id(endpoint) not in positions:
        return () if endpoint is None else (endpoint,)
    first, last = sorted((positions[id(anchor)], positions[id(endpoint)]))
    return tuple(catalog[first:last + 1])


def _membership_after_intent(catalog, selected, operands, intent):
    identities = {id(frame) for frame in selected}
    operand_ids = {id(frame) for frame in operands}
    if intent is FrameSelectionIntent.EXACT:
        identities = operand_ids
    elif intent is FrameSelectionIntent.TOGGLE_TRACE:
        identities.symmetric_difference_update(operand_ids)
    elif intent is FrameSelectionIntent.REMOVE_TRACE_RANGE:
        identities.difference_update(operand_ids)
    else:
        identities.update(operand_ids)
    return tuple(frame for frame in catalog if id(frame) in identities)


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


__all__ = ["BrowserProfile", "BrowserResult", "BrowserView", "keyed_list_profile"]
