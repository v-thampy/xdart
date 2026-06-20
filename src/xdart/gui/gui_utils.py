# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import os
import json

# Other imports
import numpy as np
import pandas as pd
from pyFAI.units import Unit

# Qt imports
import pyqtgraph as pg
from pyqtgraph import Qt
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree.ParameterItem import ParameterItem
from pyqtgraph.parametertree.Parameter import Parameter

# This module imports


def get_rect(x, y):
    """Gets a QRectF object from given x and y data.

    args:
        x, y: arrays, x and y data

    returns:
        QRectF object, or a unit rect if x or y is empty
    """
    if len(x) == 0 or len(y) == 0:
        return Qt.QtCore.QRectF(0, 0, 1, 1)
    left = x[0]
    top = y[0]
    width = max(x) - min(x)
    height = max(y) - min(y)
    return Qt.QtCore.QRectF(left, top, width, height)


class RectViewBox(pg.ViewBox):
    """Special viewbox based on pyqtgraph ViewBox. Uses a box for zoom
    functions, scroll wheel to zoom.

    Mouse bindings:
      * Left-drag       — rubber-band zoom rectangle (RectMode default).
      * Middle-click    — auto-range (zoom-out).  Bound here because the
                          right-click default was consuming the context
                          menu; middle-click is unmapped in pyqtgraph
                          defaults so it's a clean home for "zoom out".
      * Right-click     — pyqtgraph's standard context menu (View All,
                          X/Y axis options including Log Scale and
                          grid toggles, Mouse Mode, Plot Options,
                          Export...).  Previously suppressed here by a
                          custom override that intercepted right-click
                          for auto-range; restored so the user gets
                          the full pyqtgraph toolbox back.
    """
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        # Middle-click → auto-range.  Right-click falls through to
        # pyqtgraph's ViewBox.mouseClickEvent, which calls
        # raiseContextMenu(ev) when the menu is enabled.
        if ev.button() == QtCore.Qt.MiddleButton:
            ev.accept()
            self.autoRange()
            return
        pg.ViewBox.mouseClickEvent(self, ev)
            
    def mouseDragEvent(self, ev, axis=None):
        ev.accept()  ## we accept all buttons
        
        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        ## Ignore axes if mouse is disabled
        mouseEnabled = np.array(self.state['mouseEnabled'], dtype=float)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1-axis] = 0.0

        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        
        elif ev.button() == QtCore.Qt.LeftButton:
            pg.ViewBox.mouseDragEvent(self, ev)
        
        else:
            tr = dif*mask
            tr = self.mapToView(tr) - self.mapToView(Point(0,0))
            x = tr.x() if mask[0] == 1 else None
            y = tr.y() if mask[1] == 1 else None
            
            self._resetTarget()
            if x is not None or y is not None:
                self.translateBy(x=x, y=y)
            self.sigRangeChangedManually.emit(self.state['mouseEnabled'])


class DFTableModel(QtCore.QAbstractTableModel):
    """TableModel for handling pandas DataFrame. Used with a QTableView.
    See QAbstractTableModel for details on implemented methods.
    
    attributes:
        dataFrame: pandas DataFrame, where data is stored.
    """
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        if data is None:
            data = pd.DataFrame()
        self.dataFrame = data
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self.dataFrame.iloc[index.row(), index.column()])
        return None

    def rowCount(self, parent):
        return self.dataFrame.shape[0]
    
    def columnCount(self, parent):
        return self.dataFrame.shape[1]
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                try:
                    return str(self.dataFrame.columns.values[section])
                except IndexError:
                    return ' '
            elif orientation == QtCore.Qt.Vertical:
                try:
                    return str(self.dataFrame.index[section])
                except IndexError:
                    return ' '
        return QtCore.QAbstractTableModel.headerData(self, section, orientation, role)
    
    
class NamedActionParameterItem(ParameterItem):
    """pyqtgraph ActionParameterItem which can display a title.
    """
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        # self.layoutWidget = QtGui.QWidget()
        self.layoutWidget = QtWidgets.QWidget()
        # self.layout = QtGui.QHBoxLayout()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        title = param.opts.get('title', None)
        if title is None:
            title = param.name()
        # self.button = QtGui.QPushButton(title)
        self.button = QtWidgets.QPushButton(title)
        #self.layout.addSpacing(100)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.button.clicked.connect(self.buttonClicked)
        param.sigNameChanged.connect(self.paramRenamed)
        self.setText(0, '')
        
    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return
        
        self.setFirstColumnSpanned(True)
        tree.setItemWidget(self, 0, self.layoutWidget)
        
    def paramRenamed(self, param, name):
        self.button.setText(name)
        
    def buttonClicked(self):
        self.param.activate()

        
class NamedActionParameter(Parameter):
    """Used for displaying a button within the tree."""
    itemClass = NamedActionParameterItem
    sigActivated = QtCore.Signal(object)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)


from pyqtgraph.parametertree import registerParameterType
from pyqtgraph.parametertree.parameterTypes.basetypes import SimpleParameter
from pyqtgraph.parametertree.parameterTypes.str import StrParameterItem


class _TailLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that keeps the END of its text visible (cursor parked at the
    end) whenever it is not being edited — so a long path shows its tail (the
    file name) rather than the root."""

    def _park_tail(self):
        if not self.hasFocus():
            self.setCursorPosition(len(self.text()))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._park_tail()


class StrBrowseParameterItem(StrParameterItem):
    """A ``str`` (path) parameter rendered as an always-visible line edit with an
    INLINE ``Browse`` button, instead of the path and a separate full-width
    Browse row.

    The button emits the parameter's ``sigActivated`` — the same contract as
    :class:`NamedActionParameter` — so existing browse handlers wire straight to
    the path parameter (no separate ``*_browse`` action param).  The line edit
    shows the END of long paths (see :class:`_TailLineEdit`)."""

    def makeWidget(self):
        # Always show the editor + button (no label/editor swap), so Browse is
        # always reachable and the path stays editable inline.
        self.hideWidget = False
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        le = _TailLineEdit()
        # Match the wrangler tree's darker input tone.  Set explicitly because
        # this custom widget otherwise falls back to the global button/input
        # styles instead of the tree-local QLineEdit rule.
        le.setStyleSheet('border: 0px; background-color: #3f4354;')
        btn = QtWidgets.QPushButton(self.param.opts.get('buttonText', 'Browse'))
        btn.setMaximumWidth(72)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        # Tint the Browse button a distinct indigo-gray so it reads as an action
        # against the same-shade path field + neighbouring buttons (the default
        # QPushButton bg #3a3d4d matches the inputs).  Scoped here like the line
        # edit's inline style above; hover picks up the purple focus accent.
        btn.setStyleSheet(
            "QPushButton {"
            " background-color: #5269a8; color: #f8f8f2;"
            " border: 1px solid #6075b5; border-radius: 3px; padding: 1px 6px; }"
            "QPushButton:hover {"
            " background-color: #627ac0; border-color: #bd93f9; }"
            "QPushButton:pressed { background-color: #465d96; }"
        )
        lay.addWidget(le, 1)
        lay.addWidget(btn)
        btn.clicked.connect(lambda: self.param.activate())

        def _set_value(v):
            text = '' if v is None else str(v)
            le.setText(text)
            le.setToolTip(text)
            le._park_tail()

        # WidgetParameterItem contract — proxied to the line edit.
        w.sigChanged = le.editingFinished
        w.sigChanging = le.textChanged
        w.value = le.text
        w.setValue = _set_value
        w.setReadOnly = le.setReadOnly          # WidgetParameterItem may call this
        w.setFocusProxy(le)
        w.lineEdit = le
        self._lineEdit = le
        self._browseButton = btn
        return w

    def treeWidgetChanged(self):
        StrParameterItem.treeWidgetChanged(self)
        # The composite editor is always shown (hideWidget=False); make sure the
        # stand-in display label never claims space beside it.
        dl = getattr(self, 'displayLabel', None)
        if dl is not None:
            dl.hide()
        # Re-park once the row is in the tree (the widget now has a real width).
        le = getattr(self, '_lineEdit', None)
        if le is not None:
            le._park_tail()


class StrBrowseParameter(SimpleParameter):
    """A ``'str_browse'`` parameter: a string path value with an inline Browse
    button (:class:`StrBrowseParameterItem`).  Activating the button emits
    ``sigActivated`` like :class:`NamedActionParameter`, so it is a drop-in for a
    ``str`` path field + its separate ``Browse`` action."""
    itemClass = StrBrowseParameterItem
    sigActivated = QtCore.Signal(object)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)

    def _interpretValue(self, v):
        return '' if v is None else str(v)


registerParameterType('str_browse', StrBrowseParameter, override=True)


class XdartEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == Unit:
            return {
                '_type': 'pfunit',
                'value': str(o)
            }
        return super().default(o)


class XdartDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, obj):
        if '_type' not in obj:
            return obj
        if obj['_type'] == 'pfunit':
            return Unit(obj['value'])
        return obj
