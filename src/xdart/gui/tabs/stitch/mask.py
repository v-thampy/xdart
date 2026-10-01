"""Stitch mask authoring through the established exact-frame child workflow."""
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import SingleWorkerOwner, OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.widgets.controls_panel import FormRow


class StitchMaskDialog(QtWidgets.QDialog):
    maskSaved = QtCore.Signal(str)

    def __init__(self, project_root, *, source=None, rotation=0, reader_options=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make detector mask")
        self.resize(570, 190)
        self._project_root = str(project_root)
        self._source = source
        self._reader_options = dict(reader_options or ({} if source is None else source.options.get("read_image_kwargs", {})))
        self._rotation = int(rotation)
        self._closing = False
        self._worker = SingleWorkerOwner(self._run)
        self._identity = None
        layout = QtWidgets.QVBoxLayout(self)
        # Int's rows (Codex Stitch/Int reuse audit R3): the image, chosen with its 📁 and shown by name, and
        # the frame, a whole number from 0
        self._image_row = FormRow(label="Image", path=("Mask", "image"),
                                  value="" if source is None else str(source.uri), browse=True)
        self.source_label = self._image_row.editor
        self.source_label.setReadOnly(True)
        self.source_label.setToolTip("The original detector image or raw NeXus file the mask is drawn on.")
        self.choose = self._image_row.browse_button
        self._image_row.browseRequested.connect(lambda _path: self._choose())
        layout.addWidget(self._image_row)
        self._frame_row = FormRow(label="Frame", path=("Mask", "frame"), value="0", kind="int", minimum=0)
        self.frame = self._frame_row.editor
        layout.addWidget(self._frame_row)
        self.status = QtWidgets.QLabel("The selected original frame opens in detector coordinates. The mask follows the processing orientation when used.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.launch = QtWidgets.QPushButton("Open mask editor")
        self.launch.clicked.connect(self.begin)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.close)
        buttons.addWidget(self.launch)
        buttons.addWidget(self.cancel)
        layout.addLayout(buttons)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    def set_source_path(self, selected):
        from xrd_tools.core.scan import SourceSpec, SourceKind
        from xrd_tools.sources.registry import guess_source_kind
        kind = guess_source_kind(selected)
        if kind not in {SourceKind.IMAGE_FILE, SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            raise ValueError("Choose an original detector image or raw NeXus file")
        options = {key: value for key, value in self._reader_options.items()
                   if key in {"detector_shape", "detector", "raw_dtype", "raw_header_skip"}} if kind is SourceKind.IMAGE_FILE else {}
        self._source = SourceSpec(selected, kind, options=options)
        self.source_label.setText(str(selected))

    def _choose(self):
        from xdart.gui.widgets import file_dialogs
        picks = file_dialogs.browse(self, "Choose original detector image",
                                    file_filter=file_dialogs.DETECTOR_SOURCE_FILTER,
                                    current=str(getattr(self._source, "uri", "") or ""),
                                    fallback=str(self._project_root or ""))
        selected = picks[0] if picks else ""
        if selected:
            try:
                self.set_source_path(selected)
            except (TypeError, ValueError, OSError) as error:
                self.status.setText(str(error))

    @staticmethod
    def _run(request, cancelled, publish):
        from xrd_tools.integrate.stitch_calibration import capture_calibration_frames
        from xdart.gui.tabs.scattering.experiment_authoring import (
            prepare_mask_request, run_mask, mask_terminal_result_valid,
        )
        from xdart.gui.tabs.scattering.operation_values import OperationIdentity
        source, frame, folder, rotation = request
        publish("Preparing mask image", 0, 1)
        rows = capture_calibration_frames(source, (frame,), folder, rotation=0, motor_names=())
        if cancelled.is_set():
            return None
        mask_request = prepare_mask_request(rows[0].image_path)
        terminal = run_mask(mask_request, OperationIdentity(1), cancelled,
                            lambda *_args: None, lambda _identity: not cancelled.is_set())
        if not mask_terminal_result_valid(terminal, mask_request):
            raise ValueError("Mask editor did not return a qualified saved mask")
        if terminal.status.value == "cancelled":
            return None
        if terminal.status.value != "returned":
            raise ValueError(terminal.diagnostic or "Mask was not saved")
        return terminal.payload.request.final_path

    def begin(self):
        if self._source is None or not self._project_root:
            self.status.setText("Choose a Project folder and an original detector image first.")
            return
        refusal = self._frame_row.refusal()
        if refusal:
            self.status.setText(f"Frame: {refusal}")
            return
        folder = str(Path(self._project_root) / "xdart_calibration" / "masks")
        frame = int(str(self._frame_row.current_value()).strip() or 0)
        self._identity = self._worker.begin((self._source, frame, folder, self._rotation))
        if self._identity is not None:
            self.launch.setEnabled(False)
            self.choose.setEnabled(False)
            self.frame.setEnabled(False)
            self.status.setText("Preparing original detector image…")
            self._timer.start()

    def _poll(self):
        update = self._worker.poll(self._identity)
        if update is None:
            if self._closing and self._worker.close().status is PageCleanup.CLEAN:
                self._timer.stop()
                super().close()
            return
        if update.progress is not None:
            self.status.setText(update.progress.stage)
        elif update.terminal is not None:
            self._timer.stop()
            self.launch.setEnabled(not self._closing)
            self.choose.setEnabled(not self._closing)
            self.frame.setEnabled(not self._closing)
            terminal = update.terminal
            if terminal.status is OperationTerminalStatus.RETURNED and terminal.payload:
                self.status.setText(f"Saved: {terminal.payload}")
                self.maskSaved.emit(terminal.payload)
            else:
                self.status.setText(terminal.failure_message or "Mask editor closed without a saved mask")
            if self._closing:
                self.close()

    def active(self):
        return self._worker.busy

    def shutdown(self):
        self._closing = True
        receipt = self._worker.close()
        if receipt.status is PageCleanup.CLEAN:
            self._timer.stop()
        return receipt

    def closeEvent(self, event):
        receipt = self.shutdown()
        if receipt.status is PageCleanup.PENDING:
            self.status.setText("Closing mask editor…")
            self._timer.start()
            event.ignore()
        else:
            event.accept()
