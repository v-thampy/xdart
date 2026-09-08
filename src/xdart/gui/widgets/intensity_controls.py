"""Display-only intensity controls shared by the standalone viewers.

The host owns the target plot/image and displayed units.  Synchronization is
silent; only operator actions emit signals.  No numerical source is retained.
"""

from __future__ import annotations

from math import isfinite

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def _interval(values):
    if values is None:
        return None
    lo, hi = (float(value) for value in values)
    return (lo, hi) if isfinite(lo) and isfinite(hi) and hi > lo else None


class IntensityRangeSlider(QtWidgets.QWidget):
    """Canonical two-handle horizontal slider, with exact-entry activation."""

    sigRangeChanged = QtCore.Signal(float, float)
    sigEditRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin, self._vmax = 0.0, 1.0
        self._lo, self._hi = 0.0, 1.0
        self._drag = None
        self.setMinimumSize(150, 24)
        self.setMaximumHeight(28)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Manual intensity range — drag the handles, or double-click "
            "to type exact min/max values."
        )

    def values(self):
        return self._lo, self._hi

    def domain(self):
        return self._vmin, self._vmax

    def has_valid_domain(self):
        return _interval(self.domain()) is not None

    def setDomain(self, vmin, vmax, *, lower=None, upper=None, emit=False):
        domain = _interval((vmin, vmax))
        if domain is None:
            self.setEnabled(False)
            return
        changed = domain != self.domain()
        self._vmin, self._vmax = domain
        self.setEnabled(True)
        values_changed = self.setValues(
            vmin if lower is None else lower,
            vmax if upper is None else upper,
            emit=emit,
        )
        if changed and not values_changed:
            self.update()

    def setValues(self, lower, upper, *, emit=True):
        lo = min(max(float(lower), self._vmin), self._vmax)
        hi = min(max(float(upper), self._vmin), self._vmax)
        if hi < lo:
            lo, hi = hi, lo
        changed = (lo, hi) != self.values()
        if changed:
            self._lo, self._hi = lo, hi
            self.update()
            if emit:
                self.sigRangeChanged.emit(lo, hi)
        return changed

    def _track_rect(self):
        return QtCore.QRectF(9, self.height() / 2 - 3, max(1, self.width() - 18), 6)

    def _x_for_value(self, value):
        rect = self._track_rect()
        fraction = (value - self._vmin) / (self._vmax - self._vmin)
        return rect.left() + min(max(fraction, 0.0), 1.0) * rect.width()

    def _value_for_x(self, x):
        rect = self._track_rect()
        fraction = min(max((float(x) - rect.left()) / rect.width(), 0.0), 1.0)
        return self._vmin + fraction * (self._vmax - self._vmin)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self._track_rect()
        disabled = not self.isEnabled()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#282a36" if disabled else "#3a3d4d"))
        painter.drawRoundedRect(rect, 3, 3)
        lo_x, hi_x = self._x_for_value(self._lo), self._x_for_value(self._hi)
        painter.setBrush(QtGui.QColor("#6272a4" if disabled else "#bd93f9"))
        painter.drawRoundedRect(
            QtCore.QRectF(lo_x, rect.top(), max(1.0, hi_x - lo_x), rect.height()), 3, 3,
        )
        painter.setBrush(QtGui.QColor("#6272a4" if disabled else "#f8f8f2"))
        for x in (lo_x, hi_x):
            painter.drawEllipse(QtCore.QPointF(x, rect.center().y()), 5.5, 5.5)

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        x = event.position().x()
        self._drag = (
            "lo" if abs(x - self._x_for_value(self._lo))
            <= abs(x - self._x_for_value(self._hi)) else "hi"
        )
        self._move_handle(x)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._move_handle(event.position().x())

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        if self.isEnabled():
            self._drag = None
            self.sigEditRequested.emit()

    def _move_handle(self, x):
        value = self._value_for_x(x)
        if self._drag == "lo":
            self.setValues(min(value, self._hi), self._hi)
        elif self._drag == "hi":
            self.setValues(self._lo, max(value, self._lo))


class IntensityControls(QtWidgets.QFrame):
    """Intensity slider, Autoscale toggle, and exact displayed-value entry.

    ``sync(domain, current, reset=False)`` accepts finite increasing pairs or
    ``None`` for unavailable data.  Manual limits remain absolute across every
    domain change; the host may explicitly reset after a mode/unit transition.
    The public ``autoscale`` button and ``values()`` expose the current choice.
    """

    rangeChanged = QtCore.Signal(float, float)
    autoToggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewerIntensityControls")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Intensity", self))
        self.slider = IntensityRangeSlider(self)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)
        self.autoscale = QtWidgets.QPushButton("Autoscale", self)
        self.autoscale.setCheckable(True)
        self.autoscale.setChecked(True)
        self.autoscale.setFixedHeight(28)
        self.autoscale.setMinimumWidth(112)
        layout.addWidget(self.autoscale)
        self._domain = self._current = None
        self._entry_popup = None
        self.slider.sigRangeChanged.connect(self._range_changed)
        self.slider.sigEditRequested.connect(self._open_entry)
        self.autoscale.toggled.connect(self._auto_toggled)

    def values(self):
        return self.slider.values()

    def sync(self, domain, current, *, reset=False):
        domain, current = _interval(domain), _interval(current)
        current = current or domain
        self._domain, self._current = domain, current
        if reset:
            blocker = QtCore.QSignalBlocker(self.autoscale)
            self.autoscale.setChecked(True)
            del blocker
        if domain is None:
            self.slider.setEnabled(False)
            return
        self._set_window(current if self.autoscale.isChecked() else self.values())

    def _set_window(self, values):
        if self._domain is None:
            self.slider.setEnabled(False)
            return
        lo, hi = values
        self.slider.setDomain(
            min(self._domain[0], lo), max(self._domain[1], hi),
            lower=lo, upper=hi, emit=False,
        )

    def _auto_toggled(self, checked):
        if self._current is not None and self._domain is not None:
            self._set_window(self._current)
        self.autoToggled.emit(checked)

    def _range_changed(self, lo, hi):
        if not self.autoscale.isChecked():
            self.rangeChanged.emit(lo, hi)

    def _open_entry(self):
        if not self.slider.isEnabled():
            return
        if self._entry_popup is None:
            popup = QtWidgets.QFrame(self, QtCore.Qt.WindowType.Popup)
            popup.setObjectName("intensityEntryPopup")
            popup.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            layout = QtWidgets.QHBoxLayout(popup)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(6)
            for label, name in (("Min", "intensityMinimum"), ("Max", "intensityMaximum")):
                edit = QtWidgets.QLineEdit(popup)
                edit.setObjectName(name)
                edit.setFixedWidth(92)
                edit.returnPressed.connect(self._apply_entry)
                layout.addWidget(QtWidgets.QLabel(label, popup))
                layout.addWidget(edit)
            button = QtWidgets.QPushButton("Apply", popup)
            button.setObjectName("intensityApply")
            button.setDefault(True)
            button.clicked.connect(self._apply_entry)
            layout.addWidget(button)
            self._entry_popup = popup
        popup = self._entry_popup
        low, high = self._entry_editors()
        lo, hi = self.values()
        low.setText(f"{lo:g}")
        high.setText(f"{hi:g}")
        popup.move(self.slider.mapToGlobal(QtCore.QPoint(0, self.slider.height())))
        popup.show()
        low.setFocus()
        low.selectAll()

    def _entry_editors(self):
        return tuple(self._entry_popup.findChild(QtWidgets.QLineEdit, name) for name in (
            "intensityMinimum", "intensityMaximum",
        ))

    def _apply_entry(self):
        low, high = self._entry_editors()
        try:
            values = _interval((low.text(), high.text()))
        except ValueError:
            return
        if values is None or self._domain is None:
            return
        self._entry_popup.hide()
        self.autoscale.setChecked(False)
        self._set_window(values)
        self.rangeChanged.emit(*values)
