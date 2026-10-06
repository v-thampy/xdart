"""Standalone compatibility wrapper over Int's shared ROI selection owner."""
from pyqtgraph.Qt import QtCore, QtWidgets
from xdart.gui.tabs.scattering.roi_selection_view import RoiSelectionView, _rect_center_size  # noqa: F401


class RoiSelectDialog(QtWidgets.QDialog):
    sigCompute = QtCore.Signal(object)
    sigPreviewFrame = QtCore.Signal(int)

    def __init__(self, image, parent=None, *, frame_labels=(), frame_label=None):
        super().__init__(parent)
        self.setObjectName("roiSelectDialog")
        self.setWindowTitle("Select ROIs")
        self.resize(560, 680)
        self._view = RoiSelectionView(image, self, frame_labels=frame_labels, frame_label=frame_label)
        QtWidgets.QVBoxLayout(self).addWidget(self._view)
        self._view.sigCompute.connect(self.sigCompute)
        self._view.sigPreviewFrame.connect(self.sigPreviewFrame)
        self._view.closeRequested.connect(self.close)
        from xdart.gui.widgets.context_help import owner_for
        self._help_owner = owner_for(self)

    def __getattr__(self, name):
        view = self.__dict__.get("_view")
        if view is not None:
            return getattr(view, name)
        raise AttributeError(name)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)
