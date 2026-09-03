"""Lazy adapter from the Scattering Workspace to the generic page handle.

This factory constructs the real ``ScatteringWorkspace`` with
its production services and returns a typed ``PageHandle`` whose ports map the
page truthfully — Open/Run/Stop, profile persistence, slice pinning, and
write-mode changes dispatch through the page's single command owner; run
activity comes from the coordinator phase, application menus mount onto the
page's own Config/Help hosts, activity joins run and page-operation ownership,
and close uses
``close_workspace()``'s cleanup receipt (CLEANED is authoritatively CLEAN).
file-dialog choosers are ported unchanged from the live-verified opt-in
launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from .handle import AppMenuHosts, PageHandle
from .services import HostServices
from .values import (
    ActionAccepted,
    ActionCompleted,
    CloseReceipt,
    PageCleanup,
    SCATTERING_PAGE_KEY,
)

if TYPE_CHECKING:
    from pyqtgraph import QtWidgets as _QtWidgets  # noqa: F401 (typing only)
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace


def _tool_button_menu(widget, object_name: str):
    """The page-owned QMenu behind one named toolbutton, fail-loud."""
    from pyqtgraph.Qt import QtWidgets

    button = widget.findChild(QtWidgets.QToolButton, object_name)
    if button is None or button.menu() is None:
        raise RuntimeError(f"vNext menu host {object_name!r} is missing")
    return button.menu()


@dataclass(frozen=True, slots=True)
class _WorkspaceMenus:
    widget: "ScatteringWorkspace"

    def mount_points(self) -> AppMenuHosts:
        return AppMenuHosts(
            _tool_button_menu(self.widget, "configMenuButton"),
            _tool_button_menu(self.widget, "helpMenuButton"),
            _tool_button_menu(self.widget, "analysisMenuButton"),
        )


@dataclass(frozen=True, slots=True)
class _WorkspaceActivity:
    """Truthful activity from the lifecycle and page-operation owners."""

    lifecycle: object
    operation: object
    pending: object | None = None

    def active(self) -> bool:
        from xdart.gui.tabs.scattering.state_machine import RunPhase

        try:
            phase = self.lifecycle.phase
            operation_owned = self.operation.owned
            pending_owned = (
                False if self.pending is None else self.pending()
            )
        except Exception:
            return True
        if (type(operation_owned) is not bool
                or type(pending_owned) is not bool):
            return True
        return (
            phase not in (RunPhase.IDLE, RunPhase.FAILED, RunPhase.CLOSED)
            or operation_owned or pending_owned
        )


@dataclass(slots=True)
class _WorkspaceCloser:
    """Map ``close_workspace()`` onto the host close-receipt contract.

    ``close_workspace`` is re-entrant and converges: the host may re-poll a
    PENDING receipt and each call continues the same close rather than
    starting a second one. ``CleanupStatus.CLEANED`` maps authoritatively to
    ``CLEAN`` — the executor legitimately retains diagnostics from a transient
    cleanup failure that later succeeded, and those belong in the detail, not
    in the verdict (a latched terminal receipt must never livelock exit).
    """

    widget: "ScatteringWorkspace"

    def __call__(self) -> CloseReceipt:
        from xdart.gui.tabs.scattering.events import CleanupStatus

        closed = self.widget.close_workspace()
        detail = closed.cleanup_status.value
        if closed.cleanup_failures:
            detail += (
                f" ({len(closed.cleanup_failures)} retained cleanup"
                " diagnostics)"
            )
        if closed.cleanup_status is CleanupStatus.CLEANED:
            return CloseReceipt(PageCleanup.CLEAN, detail)
        return CloseReceipt(PageCleanup.PENDING, detail)


def _dispatch(widget, kind_name: str, value=None):
    """Route one host action through the page's single command owner."""
    from xdart.gui.tabs.scattering.shell_values import (
        ShellCommand,
        ShellCommandKind,
    )

    widget._handle_shell_command(
        ShellCommand(ShellCommandKind[kind_name], value))


@dataclass(frozen=True, slots=True)
class _WorkspaceOpenFolder:
    widget: "ScatteringWorkspace"

    def request(self):
        _dispatch(self.widget, "MENU", "File:Open Folder")
        return ActionCompleted("browser-directory")


@dataclass(frozen=True, slots=True)
class _WorkspaceRunControl:
    """Dispatch the exact shell Run/Stop commands; the page owns admission."""

    widget: "ScatteringWorkspace"

    def run_pause(self):
        _dispatch(self.widget, "RUN_ACTION")
        return ActionAccepted("run-action")

    def stop(self):
        _dispatch(self.widget, "STOP")
        return ActionAccepted("stop")


@dataclass(frozen=True, slots=True)
class _WorkspaceWriteMode:
    widget: "ScatteringWorkspace"

    def toggle(self):
        current = self.widget._intents.snapshot().thaw().output_mode
        value = "Append" if current == "Overwrite" else "Overwrite"
        _dispatch(self.widget, "SET_OUTPUT_POLICY", value)
        return ActionCompleted(f"write-mode-{value.casefold()}")


@dataclass(frozen=True, slots=True)
class _WorkspaceSettings:
    widget: "ScatteringWorkspace"

    def load(self):
        _dispatch(self.widget, "MENU", "Config:Load")
        return ActionCompleted("profile-load-requested")

    def save(self):
        _dispatch(self.widget, "MENU", "Config:Save")
        return ActionCompleted("profile-save-requested")


@dataclass(frozen=True, slots=True)
class _WorkspaceSlicePin:
    widget: "ScatteringWorkspace"

    def pin(self):
        _dispatch(self.widget, "PIN_SLICE")
        return ActionCompleted("slice-pin-requested")


def _control_path_chooser(widget):
    """File-dialog chooser for Controls path fields (from the live launcher)."""
    from pyqtgraph.Qt import QtWidgets

    from xdart.gui.tabs.scattering.controls_projection import (
        BACKGROUND_DIRECTORY,
        BACKGROUND_FILE,
        PONI_FILE,
        PROJECT_ROOT,
        SAVE_PATH,
        SOURCE_DIRECTORY,
    )

    def choose(path, current, start_directory):
        if path in {PROJECT_ROOT, SOURCE_DIRECTORY, BACKGROUND_DIRECTORY}:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                widget, "Choose folder", start_directory)
            return selected or None
        if path == SAVE_PATH:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                widget, "Choose processed-data folder", start_directory)
            return selected or None
        title, file_filter = (
            ("Choose Background detector image",
             "Detector images (*.tif *.tiff *.cbf *.edf *.img *.mar3450 *.raw)")
            if path == BACKGROUND_FILE
            else
            ("Choose PONI calibration", "PONI files (*.poni);;All files (*)")
            if path == PONI_FILE
            else (
                "Choose detector mask",
                "Detector masks (*.edf *.tif *.tiff *.npy);;All files (*)",
            ))
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            widget, title, start_directory, file_filter)
        return selected or None

    return choose


def _qualify_mask_hdf_dialog_selection(
    source: str, data_path: str, frame: int | None,
) -> str:
    """Return one policy-qualified HDF URL without decoding its pixels."""
    from silx.io.url import DataUrl
    from xdart.gui.tabs.scattering.experiment_authoring import (
        _mask_hdf_frame,
        _mask_source,
    )

    selected = DataUrl(
        scheme="silx", file_path=source, data_path=data_path,
        data_slice=None if frame is None else (frame,),
    ).path()
    path, exact = _mask_source(selected)
    if exact is None:
        raise ValueError("Choose one exact HDF5/NeXus dataset and frame.")
    _mask_hdf_frame(path, exact, read=False)
    return exact


def _mask_hdf_dialog_type():
    """Build the xdart-owned, metadata-only HDF frame selector type."""
    from pyqtgraph.Qt import QtCore, QtWidgets

    class _MaskHdfFrameDialog(QtWidgets.QDialog):
        def __init__(self, parent, source: str):
            super().__init__(parent)
            self._source = source
            self.selected_source = None
            self.setWindowTitle("Choose HDF5/NeXus dataset and frame")
            self.setModal(True)

            layout = QtWidgets.QVBoxLayout(self)
            source_label = QtWidgets.QLabel(source, self)
            source_label.setObjectName("maskHdfSource")
            source_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            source_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            source_label.setWordWrap(True)
            layout.addWidget(source_label)

            form = QtWidgets.QFormLayout()
            self.dataset_path = QtWidgets.QLineEdit(
                "/entry/data/data_000001", self)
            self.dataset_path.setObjectName("maskHdfDatasetPath")
            self.dataset_path.setMaxLength(4096)
            form.addRow("Dataset path", self.dataset_path)
            self.no_frame = QtWidgets.QCheckBox(
                "Dataset is one 2-D image (no frame index)", self)
            self.no_frame.setObjectName("maskHdfNoFrame")
            form.addRow("", self.no_frame)
            self.frame_index = QtWidgets.QSpinBox(self)
            self.frame_index.setObjectName("maskHdfFrameIndex")
            self.frame_index.setRange(0, 2_147_483_647)
            self.frame_index.setValue(0)
            form.addRow("Frame index", self.frame_index)
            layout.addLayout(form)
            self.no_frame.toggled.connect(
                lambda checked: self.frame_index.setEnabled(not checked))

            self.error_label = QtWidgets.QLabel("", self)
            self.error_label.setObjectName("maskHdfError")
            self.error_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            self.error_label.setWordWrap(True)
            layout.addWidget(self.error_label)
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Open
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
                parent=self,
            )
            buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Open,
            ).setText("Use frame")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def accept(self):
            data_path = self.dataset_path.text().strip()
            frame = None if self.no_frame.isChecked() \
                else int(self.frame_index.value())
            try:
                selected = _qualify_mask_hdf_dialog_selection(
                    self._source, data_path, frame)
            except Exception as exc:
                message = str(exc).strip()
                self.error_label.setText(
                    (message or "The HDF5 selection is unavailable.")[:512])
                self.selected_source = None
                return
            self.error_label.clear()
            self.selected_source = selected
            super().accept()

    return _MaskHdfFrameDialog


def _authoring_source_chooser(widget):
    """Dedicated source chooser for standalone Calibrate and Make Mask."""
    from pyqtgraph.Qt import QtWidgets

    def choose(asset, start_directory):
        if asset == "poni":
            title = "Choose calibration source image"
            file_filter = (
                "Calibration sources (*.tif *.tiff *.h5 *.hdf5 *.nxs "
                "*.nexus *.edf *.cbf *.img *.mar3450 *.raw);;All files (*)"
            )
        elif asset == "mask":
            selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
                widget, "Choose TIFF or HDF5/NeXus source for mask",
                start_directory,
                "Mask sources (*.tif *.tiff *.h5 *.hdf5 *.nxs *.nexus)",
            )
            if not selected:
                return None
            suffix = os.path.splitext(selected)[1].casefold()
            if suffix in {".tif", ".tiff"}:
                return selected
            if suffix not in {".h5", ".hdf5", ".nxs", ".nexus"}:
                return selected
            dialog = _mask_hdf_dialog_type()(widget, selected)
            result = dialog.exec()
            exact = dialog.selected_source
            dialog.deleteLater()
            if result != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            if type(exact) is not str or not exact:
                raise RuntimeError(
                    "the HDF5 selector accepted without an exact frame")
            return exact
        else:
            return None
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            widget, title, start_directory, file_filter,
        )
        return selected or None

    return choose


def _source_selection_chooser(widget):
    """Source chooser (directory/single/series) from the live launcher."""
    from pathlib import Path

    from pyqtgraph.Qt import QtWidgets

    from xrd_tools.core.scan import SourceSpec
    from xrd_tools.sources.selection import (
        DirectorySourceSpec,
        image_series_spec,
        is_single_image_spec,
        normalize_metadata_format,
        single_image_spec,
    )

    def metadata_format(source):
        if type(source) is DirectorySourceSpec:
            return normalize_metadata_format(source.metadata_format)
        if type(source) is SourceSpec:
            options = dict(source.options)
            return normalize_metadata_format(
                options.get("metadata_format", "auto"))
        return "auto"

    def choose(current, desired_mode, start_directory):
        mode = desired_mode or (
            "Image Directory"
            if type(current) is DirectorySourceSpec
            else (
                "Single Image"
                if is_single_image_spec(current)
                else "Image Series"
            )
        )
        if mode == "Image Directory":
            prior = (
                current
                if type(current) is DirectorySourceSpec
                else DirectorySourceSpec(
                    Path(),
                    suffixes=(".tif",),
                    metadata_format=metadata_format(current),
                )
            )
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                widget, "Choose image directory", start_directory)
            if not selected:
                return None
            return DirectorySourceSpec(
                Path(selected),
                recursive=prior.recursive,
                suffixes=prior.suffixes or (".tif",),
                name_filter=prior.name_filter,
                generation=prior.generation + 1,
                metadata_format=prior.metadata_format,
            )
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            widget,
            (
                "Choose one detector image"
                if mode == "Single Image"
                else "Choose an image-series member"
            ),
            start_directory,
            (
                "Detector image files (*.tif *.tiff *.cbf *.edf *.img "
                "*.mar3450 *.raw)"
                if mode == "Single Image"
                else
                "Detector images (*.tif *.tiff *.h5 *.hdf5 *.nxs *.cxi "
                "*.cbf *.edf *.img *.mar3450 *.raw);;All files (*)"
            ),
        )
        if not selected:
            return None
        if mode == "Single Image":
            return single_image_spec(
                selected, metadata_format=metadata_format(current))
        return image_series_spec(
            selected, metadata_format=metadata_format(current))

    return choose


def build_scattering_workspace(
    services: HostServices,
    parent,
) -> PageHandle:
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.adapters.run_executor import (
        StandardRunExecutor,
    )
    from xdart.gui.tabs.scattering.adapters.source import (
        FilesystemSourceAdapter,
    )
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace

    key = SCATTERING_PAGE_KEY
    intents = services.run_intents.store_for(key)
    if intents is None:
        intents = RunIntentStore(RunIntent(
            output_mode="Overwrite",
            max_cores=min(max(1, (os.cpu_count() or 1) - 1), 4),
        ))
    executor = services.execution.executor_for(key)
    if executor is None:
        executor = StandardRunExecutor()
    sources = services.sources.source_port_for(key)
    if sources is None:
        sources = FilesystemSourceAdapter()
    lifecycle = ScatteringCoordinator()

    # Dialogs parent onto the host window (exactly like the live launcher);
    # the choosers are injected through the page's constructor seam only.
    widget = ScatteringWorkspace(
        intents=intents,
        lifecycle=lifecycle,
        sources=sources,
        executor=executor,
        control_path_chooser=_control_path_chooser(parent),
        authoring_source_chooser=_authoring_source_chooser(parent),
        source_selection_chooser=_source_selection_chooser(parent),
        parent=parent,
    )

    # Operation/analysis refusal paths emit a notice and return without a
    # projection refresh; bridge them to the host status presenter so the
    # mounted application is never silent where the standalone shell spoke.
    widget.noticeChanged.connect(
        lambda text: services.status.show(str(text)))

    return PageHandle(
        key=key,
        widget=widget,
        close=_WorkspaceCloser(widget),
        open_folder=_WorkspaceOpenFolder(widget),
        settings_io=_WorkspaceSettings(widget),
        run_control=_WorkspaceRunControl(widget),
        write_mode=_WorkspaceWriteMode(widget),
        slice_pin=_WorkspaceSlicePin(widget),
        activity=_WorkspaceActivity(
            lifecycle, widget._workspace_operations,
            widget._experiment_operation_busy,
        ),
        app_menus=_WorkspaceMenus(widget),
    )
