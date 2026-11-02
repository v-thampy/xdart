"""Key policy shared by retained tool dialogs whose setup must survive Escape."""

from pyqtgraph.Qt import QtCore


class RetainedDialogKeyPolicy:
    """Mount before QDialog; editors and focused buttons handle their keys first.

    Escape never rejects the retained window. Return/Enter propagated by an editor
    must not activate QDialog's default button after committing the field. Other
    keys continue through the normal dialog handler.
    """

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)
