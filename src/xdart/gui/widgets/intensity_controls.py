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
    """Canonical two-handle horizontal slider."""

    sigRangeChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin, self._vmax = 0.0, 1.0
        self._lo, self._hi = 0.0, 1.0
        self._drag = None
        self.setMinimumSize(150, 24)
        self.setMaximumHeight(28)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Manual intensity range — drag the handles, or click Intensity "
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
        self.setToolTip(
            f"Intensity: {lo:.8g} to {hi:.8g}\n"
            "Drag handles or click Intensity to enter exact minimum and maximum."
        )
        if changed:
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

    def _move_handle(self, x):
        value = self._value_for_x(x)
        if self._drag == "lo":
            self.setValues(min(value, self._hi), self._hi)
        elif self._drag == "hi":
            self.setValues(self._lo, max(value, self._lo))


class IntensityControls(QtWidgets.QFrame):
    """Intensity slider, Autoscale toggle, and exact displayed-value entry.

    ``sync(domain, current, reset=False)`` accepts finite increasing pairs or
    ``None`` for unavailable data. Optional percentiles make the selected pair
    relative to each image's population; sync still receives absolute display
    limits for the tooltip. Absolute manual limits remain literal across data
    changes; the host may explicitly reset after a mode/unit transition.
    The public ``autoscale`` button and ``values()`` expose the current choice.
    """

    rangeChanged = QtCore.Signal(float, float)
    autoToggled = QtCore.Signal(bool)

    def __init__(self, parent=None, *, percentiles=None):
        super().__init__(parent)
        self.setObjectName("viewerIntensityControls")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        self.label = QtWidgets.QLabel("Intensity", self)
        self.label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.label.setToolTip("Double-click to enter intensity minimum and maximum.")
        self.label.installEventFilter(self)
        layout.addWidget(self.label)
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
        self._percentile_defaults = self._percentiles = None
        self._mode = "Absolute"
        self._entry_popup = None
        self.slider.sigRangeChanged.connect(self._range_changed)
        self.autoscale.toggled.connect(self._auto_toggled)
        self.autoscale.setToolTip("Use the automatic intensity limits for the current display.")
        self.set_percentile_defaults(percentiles)

    def eventFilter(self, watched, event):
        if (watched is getattr(self, "_entry_popup", None) and event.type() == QtCore.QEvent.Type.MouseButtonPress
                and not watched.rect().contains(event.position().toPoint())):
            watched.hide()
            return True
        if (watched is self.label
                and event.type() == QtCore.QEvent.Type.MouseButtonDblClick
                and event.button() == QtCore.Qt.MouseButton.LeftButton
                and self.label.rect().contains(event.position().toPoint())):
            self._open_entry()
            return True
        return super().eventFilter(watched, event)

    def values(self):
        return self._percentiles if self._mode == "Percentile" else self.slider.values()

    def percentile_values(self):
        """Selected percentile pair, or None for the existing absolute display units."""
        return self.values() if self._mode == "Percentile" else None

    def set_percentile_defaults(self, pair):
        """Enable percentile entry; update automatic defaults while retaining manual choices.

        None restores absolute-only controls for a 1-D host.
        """
        enabled = self._percentile_defaults is not None
        if pair is None:
            self._percentile_defaults = None
            self._mode = "Absolute"
            return
        pair = _interval(pair)
        if pair is None or pair[0] < 0 or pair[1] > 100:
            raise ValueError("percentile limits need 0 <= minimum < maximum <= 100")
        self._percentile_defaults = pair
        if not enabled:
            self._mode = "Percentile"
        if self.autoscale.isChecked() or self._percentiles is None:
            self._percentiles = pair
        if self._mode == "Percentile" and self._domain is not None:
            self._set_window(self._percentiles)

    def sync(self, domain, current, *, reset=False):
        domain, current = _interval(domain), _interval(current)
        current = current or domain
        self._domain, self._current = domain, current
        if reset:
            blocker = QtCore.QSignalBlocker(self.autoscale)
            self.autoscale.setChecked(True)
            del blocker
            self._percentiles = self._percentile_defaults
        if domain is None:
            self.slider.setEnabled(False)
            return
        if self._mode == "Percentile":
            self._set_window(self._percentiles)
        else:
            self._set_window(current if self.autoscale.isChecked() else self.values())

    def _set_window(self, values):
        if self._domain is None:
            self.slider.setEnabled(False)
            return
        lo, hi = values
        # Auto shows the active display interval, not the detector/data span.
        # Manual editing keeps that span and can widen it for absolute limits.
        domain = (0.0, 100.0) if self._mode == "Percentile" else values if self.autoscale.isChecked() else (
            min(self._domain[0], lo), max(self._domain[1], hi))
        self.slider.setDomain(
            *domain,
            lower=lo, upper=hi, emit=False,
        )
        absolute = self._current
        tip = (f"Absolute display limits: {absolute[0]:.8g} to {absolute[1]:.8g}\n"
               if absolute is not None else "")
        tip += "Double-click Intensity to edit limits."
        self.label.setToolTip(tip)
        self.slider.setToolTip(self.slider.toolTip() + "\n" + tip)

    def _auto_toggled(self, checked):
        if self._current is not None and self._domain is not None:
            if self._mode == "Percentile":
                if checked:
                    self._percentiles = self._percentile_defaults
                self._set_window(self._percentiles)
            else:
                self._set_window(self._current)
        self.autoToggled.emit(checked)

    def _range_changed(self, lo, hi):
        if self._mode == "Percentile":
            self._percentiles = (lo, hi)
        if self.autoscale.isChecked():
            blocker = QtCore.QSignalBlocker(self.autoscale)
            self.autoscale.setChecked(False)
            del blocker
            self.autoToggled.emit(False)
        self.rangeChanged.emit(lo, hi)

    def _open_entry(self):
        if not self.slider.isEnabled():
            return
        if self._entry_popup is None:
            popup = QtWidgets.QFrame(self, QtCore.Qt.WindowType.Popup)
            popup.setObjectName("intensityEntryPopup")
            popup.installEventFilter(self)
            popup.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            layout = QtWidgets.QHBoxLayout(popup)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(6)
            mode = QtWidgets.QComboBox(popup)
            mode.setObjectName("intensityMode")
            mode.addItems(("Percentile", "Absolute"))
            mode.setToolTip("Percentile follows each image's intensity population; Absolute keeps literal display limits.")
            mode.currentTextChanged.connect(self._change_mode)
            layout.addWidget(mode)
            for label, name in (("Min", "intensityMinimum"), ("Max", "intensityMaximum")):
                edit = QtWidgets.QLineEdit(popup)
                edit.setObjectName(name)
                edit.setFixedWidth(92)
                edit.setToolTip(f"{label}imum intensity limit in the selected Percentile or Absolute display units.")
                edit.returnPressed.connect(self._apply_entry)
                caption = QtWidgets.QLabel(label, popup)
                caption.setToolTip(edit.toolTip())
                layout.addWidget(caption)
                layout.addWidget(edit)
            button = QtWidgets.QPushButton("Apply", popup)
            button.setObjectName("intensityApply")
            button.setDefault(True)
            button.setToolTip("Apply these display limits; keep this popup open for further edits. Esc closes it.")
            button.clicked.connect(self._apply_entry)
            layout.addWidget(button)
            self._entry_popup = popup
        popup = self._entry_popup
        mode = popup.findChild(QtWidgets.QComboBox, "intensityMode")
        blocker = QtCore.QSignalBlocker(mode)
        mode.setCurrentText(self._mode)
        mode.setVisible(self._percentile_defaults is not None)
        del blocker
        low, high = self._entry_editors()
        lo, hi = self.values()
        low.setText(f"{lo:g}")
        high.setText(f"{hi:g}")
        popup.move(self.slider.mapToGlobal(QtCore.QPoint(0, self.slider.height())))
        popup.show()
        low.setFocus()
        low.selectAll()

    def _change_mode(self, mode):
        if mode == self._mode or self._percentile_defaults is None:
            return
        if self._mode == "Percentile":
            self._percentiles = self.values()
        self._mode = mode
        values = self._percentiles if mode == "Percentile" else self._current
        if values is None:
            return
        blocker = QtCore.QSignalBlocker(self.autoscale)
        self.autoscale.setChecked(False)
        del blocker
        self._set_window(values)
        for editor, value in zip(self._entry_editors(), values):
            editor.setText(f"{value:g}")
        self.rangeChanged.emit(*values)

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
        if self._mode == "Percentile" and (values[0] < 0 or values[1] > 100):
            return
        if self._mode == "Percentile":
            self._percentiles = values
        self.autoscale.setChecked(False)
        self._set_window(values)
        self.rangeChanged.emit(*values)
