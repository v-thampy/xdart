"""Current three-column scattering shell with one reconciliation boundary."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.session.readiness import (
    ControlsProjection,
    SectionId,
)

from xdart.gui.widgets.controls_panel import ControlsPanel
from xdart.gui.widgets.run_controls import RunControlsBar

from .browser_view import BrowserView
from .scientific_view import ScientificView
from .shell_widgets import apply_vnext_controls_readiness
from .shell_values import (
    RunStripProjection,
    ShellCommand,
    ShellCommandKind,
    ShellPhase,
    ShellProjection,
)
from .tools_view import ToolsView


class _HugWidthScrollArea(QtWidgets.QScrollArea):
    """A vertical-only scroll surface whose minimum width tracks its widget.

    LV-UI-7: the controls column previously carried three disagreeing width
    constants (column permitted 306, panel hint 313, stale root pin 360), so
    the splitter could allocate less than the panel truly needed and the
    right edge clipped behind a horizontal scrollbar at 1440x900.  Deriving
    the minimum here — inner widget's ``minimumSizeHint`` plus frame and
    vertical-scrollbar allowance — leaves ONE owner (the panel's layout) and
    makes horizontal overflow impossible by construction: the splitter can
    shrink plots, never the controls below their real minimum.
    """

    def minimumSizeHint(self) -> QtCore.QSize:
        base = super().minimumSizeHint()
        inner = self.widget()
        if inner is None:
            return base
        width = (
            inner.minimumSizeHint().width()
            + 2 * self.frameWidth()
            + self.verticalScrollBar().sizeHint().width()
        )
        return QtCore.QSize(width, base.height())

    def eventFilter(self, obj, event) -> bool:
        # QScrollArea absorbs the inner widget's LayoutRequest (scrolling
        # normally hides growth), so a repopulated panel never invalidates
        # the CACHED minimum the splitter distributes by — the fix above
        # would compute the right floor that nothing ever re-reads.  Forward
        # the invalidation explicitly.
        handled = super().eventFilter(obj, event)
        if (
            obj is self.widget()
            and event.type() == QtCore.QEvent.Type.LayoutRequest
        ):
            self.updateGeometry()
        return handled


class ScatteringWorkspaceShell(QtWidgets.QWidget):
    """A passive view: immutable state in, typed operator commands out."""

    commandRequested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("scatteringWorkspaceShell")
        self._reconciling = False
        self._revision = -1
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        self.splitter.setObjectName("e3MainSplitter")
        self.left = self._make_left_column()
        self.scientific = ScientificView()
        self.right = self._make_right_column()
        self.splitter.addWidget(self.left)
        self.splitter.addWidget(self.scientific)
        self.splitter.addWidget(self.right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([289, 868, 323])
        layout.addWidget(self.splitter)
        self.browser.commandRequested.connect(self._forward)
        self.tools.commandRequested.connect(self._forward)
        self.scientific.commandRequested.connect(self._forward)
        self._connect_controls()

    def _make_left_column(self) -> QtWidgets.QFrame:
        column = QtWidgets.QFrame()
        column.setObjectName("e3BrowserColumn")
        column.setMinimumWidth(255)
        layout = QtWidgets.QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.browser = BrowserView()
        self.tools = ToolsView()
        layout.addWidget(self.browser, 1)
        layout.addWidget(self.tools, 0)
        return column

    def _make_right_column(self) -> QtWidgets.QFrame:
        column = QtWidgets.QFrame()
        column.setObjectName("e3ControlsColumn")
        # LV-UI-7: no hand-pinned column minimum — it disagreed with the
        # panel's real need (306 permitted vs 313 required vs a stale 360
        # root pin), which is exactly how the 1440x900 clip happened.  The
        # width-hugging scroll area below derives the floor from the panel's
        # own minimumSizeHint, so the splitter can never starve the controls
        # and horizontal overflow is impossible by construction.
        layout = QtWidgets.QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.control_scroll = _HugWidthScrollArea()
        self.control_scroll.setObjectName("e3ControlsScroll")
        self.control_scroll.setWidgetResizable(True)
        self.controls = ControlsPanel(
            experiment_title="Configuration",
            show_section_numbers=False,
        )
        self.control_scroll.setWidget(self.controls)
        self.run_controls = RunControlsBar()
        layout.addWidget(self.control_scroll, 1)
        layout.addWidget(self.run_controls, 0)
        return column

    def _connect_controls(self) -> None:
        self.controls.fieldValueChanged.connect(
            lambda path, value: self._forward(
                ShellCommand(
                    ShellCommandKind.CONTROL_EDIT,
                    _coerce_scalar(value),
                    tuple(path),
                )
            )
        )
        self.controls.fieldDraftChanged.connect(
            lambda path, value: self._forward(
                ShellCommand(
                    ShellCommandKind.CONTROL_DRAFT,
                    _coerce_scalar(value),
                    tuple(path),
                )
            )
        )
        self.controls.fieldBrowseRequested.connect(
            lambda path: self._forward(
                ShellCommand(
                    ShellCommandKind.CONTROL_BROWSE,
                    path=tuple(path),
                )
            )
        )
        self.controls.controlActionRequested.connect(
            lambda action: self._forward(
                ShellCommand(
                    ShellCommandKind.CONTROL_ACTION,
                    str(getattr(action, "value", action)),
                )
            )
        )
        self.run_controls.modeChanged.connect(
            lambda value: self._forward(
                ShellCommand(
                    ShellCommandKind.SET_PROCESSING_MODE, value
                )
            )
        )
        self.run_controls.batchToggled.connect(
            lambda value: self._forward(
                ShellCommand(ShellCommandKind.SET_BATCH, bool(value))
            )
        )
        self.run_controls.liveToggled.connect(
            lambda value: self._forward(
                ShellCommand(ShellCommandKind.SET_LIVE, bool(value))
            )
        )
        self.run_controls.actionClicked.connect(
            lambda: self._forward(
                ShellCommand(ShellCommandKind.RUN_ACTION)
            )
        )
        self.run_controls.stopClicked.connect(
            lambda: self._forward(ShellCommand(ShellCommandKind.STOP))
        )
        self.run_controls.writeModeChanged.connect(
            lambda value: self._forward(
                ShellCommand(ShellCommandKind.SET_OUTPUT_POLICY, value)
            )
        )
        self.run_controls.coresSpin.valueChanged.connect(
            lambda value: self._forward(
                ShellCommand(ShellCommandKind.SET_CORES, int(value))
            )
        )

    def apply_state(
        self,
        state: ShellProjection,
        *,
        preserve_display: bool = False,
        preserve_scientific: bool = False,
        replace_scientific_on_failure: bool = False,
    ) -> None:
        """Reconcile passive state with independently retainable plot paint."""

        if type(state) is not ShellProjection:
            raise TypeError("shell state must be an exact ShellProjection")
        if state.revision < self._revision:
            return
        self._reconciling = True
        try:
            if preserve_scientific:
                self.scientific.reconcile_action_availability(
                    state.scientific
                )
            if not preserve_display:
                self.browser.reconcile(
                    state.browser,
                    state.navigation,
                    plot_mode=state.scientific.plot_mode,
                )
                # Qt keeps the browser cursor/highlight responsive during a
                # rapid frame gesture, but the scientific surface is
                # intentionally continuous-update=False: do not repaint an
                # older committed selection while the browser owns newer
                # pending intent.  The one trailing typed command reconciles
                # both surfaces together.
                if (
                    not preserve_scientific
                    and not self.browser.frame_selection_pending
                ):
                    current = state.navigation.current
                    local_frames = (
                        tuple(
                            frame
                            for frame in state.navigation.frames
                            if frame.artifact == current.artifact
                        )
                        if current is not None
                        else ()
                    )
                    artifact_progress = (
                        None
                        if current is None
                        else state.progress.for_artifact(current.artifact)
                    )
                    retained_index = next(
                        (
                            index
                            for index, frame in enumerate(local_frames, 1)
                            if frame is current
                        ),
                        0,
                    )
                    local_completed = (
                        max(
                            0,
                            min(
                                artifact_progress.total,
                                artifact_progress.published
                                - len(local_frames)
                                + retained_index,
                            ),
                        )
                        if artifact_progress is not None
                        else retained_index
                    )
                    local_total = (
                        artifact_progress.total
                        if artifact_progress is not None
                        else len(local_frames)
                    )
                    try:
                        self.scientific.reconcile(
                            state.scientific,
                            state.navigation,
                            completed=local_completed,
                            total=local_total,
                            detail=state.progress.detail,
                        )
                    except Exception:
                        if replace_scientific_on_failure:
                            self.replace_scientific_with_blank()
                        raise
            controls = _shell_controls(state.controls)
            self.controls.reconcile(controls)
            apply_vnext_controls_readiness(
                self.controls,
                state.controls_readiness,
            )
            self._reconcile_run_strip(state.run)
            self._revision = state.revision
        finally:
            self._reconciling = False

    def replace_scientific_with_blank(self) -> ScientificView:
        """Abandon a partially-mutated paint as one whole view object."""

        old = self.scientific
        sizes = self.splitter.sizes()
        fresh = ScientificView()
        fresh.commandRequested.connect(self._forward)
        index = self.splitter.indexOf(old)
        if index != 1:
            fresh.commandRequested.disconnect(self._forward)
            fresh.deleteLater()
            raise RuntimeError("scientific view left its exact splitter slot")
        replaced = self.splitter.replaceWidget(index, fresh)
        if replaced is not old:
            fresh.commandRequested.disconnect(self._forward)
            fresh.deleteLater()
            raise RuntimeError("scientific view replacement lost ownership")
        self.scientific = fresh
        self.splitter.setStretchFactor(1, 3)
        if sizes:
            self.splitter.setSizes(sizes)
        try:
            old.commandRequested.disconnect(self._forward)
        except (RuntimeError, TypeError):
            pass
        try:
            old.clear_workspace()
        except Exception:
            pass
        old.setParent(None)
        old.deleteLater()
        return fresh

    def _reconcile_run_strip(self, state: RunStripProjection) -> None:
        blockers = [
            QtCore.QSignalBlocker(widget)
            for widget in (
                self.run_controls,
                self.run_controls.modeCombo,
                self.run_controls.batchButton,
                self.run_controls.liveButton,
                self.run_controls.writeModeButton,
                self.run_controls.coresSpin,
            )
        ]
        self.run_controls.apply_profile(modes=state.modes)
        disabled = dict(state.disabled_modes)
        model = self.run_controls.modeCombo.model()
        for row in range(self.run_controls.modeCombo.count()):
            label = self.run_controls.modeCombo.itemText(row)
            reason = disabled.get(label, "")
            item = getattr(model, "item", lambda _row: None)(row)
            if item is not None:
                item.setEnabled(not reason)
            self.run_controls.modeCombo.setItemData(
                row,
                reason or None,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        index = self.run_controls.modeCombo.findText(state.mode)
        if index >= 0:
            self.run_controls.modeCombo.setCurrentIndex(index)
        self.run_controls.batchButton.setChecked(state.batch)
        self.run_controls.coresSpin.setMaximum(max(1, state.max_cores))
        self.run_controls.coresSpin.setValue(state.cores)
        self.run_controls.liveButton.setChecked(state.live)
        self.run_controls.set_write_mode(state.output_policy)
        self.run_controls.set_readiness_summary(
            state.readiness,
            ready=state.ready,
            tooltip=state.readiness_tooltip,
            live=state.live,
        )
        phase = {
            # Admission owns the pending Run action already. Morph immediately
            # to the eventual Pause affordance, but keep it disabled until the
            # executor has accepted the run and Pause can be honored.
            ShellPhase.PREPARING: "running",
            ShellPhase.RUNNING: "running",
            ShellPhase.PAUSING: "pausing",
            ShellPhase.PAUSED: "paused",
        }.get(state.phase, "idle")
        self.run_controls.set_action_phase(phase)
        self.run_controls.startButton.setEnabled(state.run_enabled)
        self.run_controls.set_stop_enabled(state.stop_enabled)
        locked = state.phase not in {
            ShellPhase.IDLE,
            ShellPhase.FAILED,
        }
        self.run_controls.set_mode_row_enabled(not locked)
        del blockers

    def _forward(self, command: ShellCommand) -> None:
        if not self._reconciling:
            self.commandRequested.emit(command)


def _coerce_scalar(value: object):
    if value is None or type(value) in {str, int, float, bool}:
        return value
    return str(value)


def _shell_controls(
    state: ControlsProjection,
) -> ControlsProjection:
    fields = tuple(
        field
        for field in state.fields
        if not (
            field.section is SectionId.PROJECT
            and field.path == ("Project", "output_mode")
        )
    )
    if len(fields) == len(state.fields):
        return state
    return ControlsProjection(
        state.processing_page,
        fields,
        state.section_actions,
        state.detector_summary,
    )


__all__ = ["ScatteringWorkspaceShell"]
