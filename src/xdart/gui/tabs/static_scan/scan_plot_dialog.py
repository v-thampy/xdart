# -*- coding: utf-8 -*-
"""Scan Plot popup (Direction-A Tools) — step 1: metadata plotting.

Plot per-frame scan metadata: any column vs any column, several overlaid to
compare, with an optional normalization column (y / norm).  NOT tied to the
loaded ``.nxs`` — a source picker opens any "scan" the headless source layer
classifies (processed NeXus / Eiger / TIFF-or-RAW sequence / SPEC) via
``xrd_tools.sources``; the per-frame columns come from
``xrd_tools.io.read_scan_data`` (processed NeXus) or the source's own metadata.

Step 1 = metadata only.  The "Plot ROI" path (ROI stats as computed columns) is
a later increment — see docs/design/design_scan_plotter_metadata_roi_jun2026.md.
"""

import logging
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.sources.probe import probe_first_frame, raw_is_reachable  # noqa: F401

from .peak_fit_util import CURVE_PENS
from .plot_axes import add_right_series, attach_right_axis

logger = logging.getLogger(__name__)


def _table_from_source(src):
    """Assemble a per-frame ``{column: float ndarray}`` from a FrameSource:
    ``frame_index`` + whole-array ``motors`` + per-frame ``metadata_for`` (sidecar
    dict[str, float]).  Missing/non-numeric values are NaN."""
    out = {}
    try:
        idxs = list(src.frame_indices)
    except Exception:
        return out
    if not idxs:
        return out
    out["frame_index"] = np.asarray([float(i) for i in idxs], dtype=float)
    motors = getattr(src, "motors", None) or {}
    for name, vals in motors.items():
        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == len(idxs):
            out[str(name)] = arr
    rows = []
    for i in idxs:
        try:
            rows.append(dict(src.metadata_for(i)))
        except Exception:
            rows.append({})
    keys = []
    for md in rows:
        for k in md:
            if k not in keys and k not in out:
                keys.append(k)
    for k in keys:
        col = np.full(len(idxs), np.nan, dtype=float)
        for r, md in enumerate(rows):
            try:
                col[r] = float(md.get(k))
            except (TypeError, ValueError):
                pass
        out[k] = col
    return out


def _checkable_item(text):
    """An unchecked, user-checkable ``QListWidgetItem`` for a column selector."""
    item = QtWidgets.QListWidgetItem(text)
    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    return item


def _numeric_columns(table):
    """Plottable columns: 1-D, numeric, finite-bearing."""
    out = []
    for name, arr in table.items():
        a = np.asarray(arr)
        if a.ndim == 1 and a.dtype.kind in "fiu" and np.any(np.isfinite(a)):
            out.append(name)
    return out


class ScanPlotDialog(QtWidgets.QDialog):
    """Plot scan metadata columns vs each other, overlaid, with normalization."""

    def __init__(self, default_uri=None, mask_provider=None, lock_provider=None,
                 parent=None):
        super().__init__(parent)
        self._table = {}
        self._columns = []
        self._positioner_names = []         # scanned-motor names (X-default hint)
        # ROI path: the opened raw source + its computed columns.
        # ``mask_provider(uri)`` returns the static detector mask to apply to ROI
        # stats for that source (e.g. the loaded scan's global_mask when the
        # picked source IS the loaded scan), or None.
        self._mask_provider = mask_provider
        # ``lock_provider(uri)`` returns the writer-coordinating file_lock when
        # the picked source IS the loaded (possibly live-writing) scan's .nxs,
        # else None — the dialog's .nxs reads enter it so they cannot race the
        # live writer's `r+` saves (the display readers' _locked_scan_read rule).
        self._lock_provider = lock_provider
        self._source = None
        self._source_uri = None
        self._scan_identity = None          # (uri, scan, entry) of the loaded scan
        self._raw_reachable = False
        self._first_image = None            # cached first frame (probe → picker)
        self._roi_dialog = None
        self._roi_worker = None
        self._roi_run_columns = []          # column names filling this run
        self._roi_row_of = {}               # frame_index -> table row
        self._closing = False
        self.setObjectName("scanPlotDialog")
        self.setWindowTitle("Scan Plot")
        self.resize(640, 620)
        self._build_ui()
        self._roi_redraw_timer = QtCore.QTimer(self)
        self._roi_redraw_timer.setSingleShot(True)
        self._roi_redraw_timer.setInterval(50)
        self._roi_redraw_timer.timeout.connect(self._redraw)
        if default_uri:
            self.load_uri(default_uri)

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Source — the shared scan-source widget (File/Directory, kind-adaptive,
        # SPEC scan + optional images + raw-reachable gating).
        from .scan_source_widget import ScanSourceWidget
        self.source_widget = ScanSourceWidget(mode="roi", parent=self)
        lay.addWidget(self.source_widget)

        # Axes row: X / Normalize combos (Y is the multi-select below).
        axes_row = QtWidgets.QHBoxLayout()
        axes_row.setSpacing(7)
        axes_row.addWidget(QtWidgets.QLabel("X"))
        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.setMinimumWidth(120)
        axes_row.addWidget(self.x_combo)
        axes_row.addWidget(QtWidgets.QLabel("Normalize"))
        self.norm_combo = QtWidgets.QComboBox()
        self.norm_combo.setMinimumWidth(110)
        self.norm_combo.setToolTip("Divide Y by this column (None = off)")
        axes_row.addWidget(self.norm_combo)
        axes_row.addStretch(1)
        self.roi_btn = QtWidgets.QPushButton("Plot ROI…")
        self.roi_btn.setObjectName("scanPlotRoiBtn")
        self.roi_btn.setEnabled(False)
        self.roi_btn.setToolTip(
            "Reduce rectangular ROIs over the scan's raw frames and add each as "
            "a plotted column (enabled only when the raw frames are reachable)")
        axes_row.addWidget(self.roi_btn)
        self.save_btn = QtWidgets.QPushButton("Save CSV…")
        self.save_btn.setEnabled(False)
        axes_row.addWidget(self.save_btn)
        self.log_btn = QtWidgets.QPushButton("Log")
        self.log_btn.setCheckable(True)
        self.log_btn.setToolTip("Log scale on the (left) Y axis")
        axes_row.addWidget(self.log_btn)
        lay.addLayout(axes_row)

        # Y multi-select (check several to overlay) + the plot, side by side.
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(7)
        y_col = QtWidgets.QVBoxLayout()
        y_col.setSpacing(2)
        y_col.addWidget(QtWidgets.QLabel("Y (overlay)"))
        self.y_list = QtWidgets.QListWidget()
        self.y_list.setObjectName("scanPlotYList")
        self.y_list.setMaximumWidth(160)
        self.y_list.setToolTip("Check one or more columns to plot vs X")
        y_col.addWidget(self.y_list, 1)
        y_col.addWidget(QtWidgets.QLabel("Right axis"))
        self.r_list = QtWidgets.QListWidget()
        self.r_list.setObjectName("scanPlotRList")
        self.r_list.setMaximumWidth(160)
        self.r_list.setMaximumHeight(110)
        self.r_list.setToolTip(
            "Check a (plotted) column here to draw it against a second Y axis on "
            "the right — for columns of very different magnitude")
        y_col.addWidget(self.r_list)
        body.addLayout(y_col)
        self.plot = pg.PlotWidget()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        # ~50% larger than pyqtgraph's ~9pt legend default (readability).
        self.legend.setLabelTextSize("13pt")
        self.right_vb, self.right_axis = attach_right_axis(self.plot)
        body.addWidget(self.plot, 1)
        lay.addLayout(body, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.x_combo.currentIndexChanged.connect(self._redraw)
        self.norm_combo.currentIndexChanged.connect(self._redraw)
        self.y_list.itemChanged.connect(self._redraw)
        self.r_list.itemChanged.connect(self._redraw)
        self.source_widget.sigSourceChanged.connect(self._on_source_selected)
        self.roi_btn.clicked.connect(self._open_roi_dialog)
        self.save_btn.clicked.connect(self._save_csv)
        self.log_btn.toggled.connect(self._redraw)

    # ---- source loading -------------------------------------------------
    def load_uri(self, uri):
        """Programmatically load a scan into the source widget (e.g. the dialog's
        default scan); the widget emits ``sigSourceChanged`` → loads the table."""
        self.source_widget.set_uri(uri)

    @staticmethod
    def _selection_identity(selection):
        """The (file, scan, entry) identity of a selection — unchanged when only
        the raw/image pairing was edited; ``None`` for an anonymous source."""
        if selection.spec is None:
            return None
        opts = dict(getattr(selection.spec, "options", {}) or {})
        return (str(selection.spec.uri), opts.get("scan"),
                getattr(selection.spec, "entry", None))

    def _on_source_selected(self, selection):
        """The ScanSourceWidget chose a source.  A NEW scan rebuilds the table
        wholesale (and aborts any in-flight ROI run); when only the raw/image
        pairing of the SAME scan changed, refresh the source + ROI gating WITHOUT
        wiping the table or the user's computed ROI columns."""
        if selection is None or selection.source is None:
            self._abort_roi_run()
            self._source = None
            self._source_uri = None
            self._raw_reachable = False
            self._first_image = None
            self._scan_identity = None
            self._positioner_names = []
            self.set_table("", {})
            self._update_roi_button()
            return
        identity = self._selection_identity(selection)
        same_scan = identity is not None and identity == self._scan_identity
        self._source = selection.source
        self._source_uri = str(selection.spec.uri) if selection.spec else None
        self._raw_reachable = bool(selection.reachable)
        self._first_image = selection.first_image
        if same_scan:
            self._update_roi_button()       # only re-gate; keep table + ROI columns
            return
        self._abort_roi_run()
        self._scan_identity = identity
        self._positioner_names = self._positioners_for(selection)
        label, table = self._table_for(selection)
        self.set_table(label, table)
        self._update_roi_button()

    def _read_locked(self, uri):
        """Context for a ``.nxs`` read of ``uri`` — the provider's
        writer-coordinating lock when ``uri`` is the loaded scan's data file,
        else a no-op."""
        from contextlib import nullcontext
        lock = None
        if self._lock_provider is not None:
            try:
                lock = self._lock_provider(str(uri))
            except Exception:
                logger.debug("scan-plot: lock provider failed", exc_info=True)
                lock = None
        return lock if lock is not None else nullcontext()

    def _table_for(self, selection):
        """Per-frame metadata table for a selection — the full ``scan_data`` for a
        processed NeXus, else the source's own per-frame metadata (which is just
        ``frame_index`` for a metadata-less image stack)."""
        from xrd_tools.core.scan import SourceKind
        spec, source = selection.spec, selection.source
        table = {}
        if spec is not None and spec.kind is SourceKind.PROCESSED_NEXUS:
            from xrd_tools.io import read_scan_data
            try:
                with self._read_locked(spec.uri):
                    table = read_scan_data(spec.uri)
            except Exception:
                logger.exception("scan-plot: read_scan_data failed")
                table = {}
        if not table:
            table = _table_from_source(source)
        return selection.label, table

    def _positioners_for(self, selection):
        """Scanned-motor names for the X default.  For a processed NeXus these
        are the recorded NXpositioner motors (``get_metadata`` — intentionally
        narrow to the diffractometer/scanned motors); for other sources they are
        the source's own ``motors`` keys.  Best-effort: a read failure yields no
        hint (X then falls back to ``frame_index``)."""
        from xrd_tools.core.scan import SourceKind
        spec, source = selection.spec, selection.source
        if spec is not None and spec.kind is SourceKind.PROCESSED_NEXUS:
            from xrd_tools.io import get_metadata
            try:
                with self._read_locked(spec.uri):
                    positioners = get_metadata(spec.uri).get("positioners") or {}
                return [str(k) for k in positioners]
            except Exception:
                logger.exception("scan-plot: reading positioners failed")
                return []
        motors = getattr(source, "motors", None) or {}
        return [str(k) for k in motors]

    def _abort_roi_run(self):
        """Stop an in-flight ROI computation + close the picker and forget its
        run state — used when the source is swapped (the old columns vanish with
        the replaced table, so unlike ``closeEvent`` we clear ``_roi_run_columns``
        here rather than leaving them for ``_on_roi_done`` to drop)."""
        if self._roi_worker is not None and self._roi_worker.isRunning():
            self._roi_worker.stop()
        self._roi_run_columns = []
        self._roi_row_of = {}
        if self._roi_dialog is not None:
            self._roi_dialog.close()
            self._roi_dialog = None

    def set_table(self, label, table):
        """Load a per-frame column table + populate the X/Y/Normalize selectors."""
        self._table = dict(table or {})
        self._columns = _numeric_columns(self._table)
        cols = self._columns

        self.x_combo.blockSignals(True)
        self.norm_combo.blockSignals(True)
        self.y_list.blockSignals(True)
        self.r_list.blockSignals(True)
        self.x_combo.clear()
        self.norm_combo.clear()
        self.y_list.clear()
        self.r_list.clear()
        self.x_combo.addItems(cols)
        self.norm_combo.addItem("None", None)
        for c in cols:
            self.norm_combo.addItem(c, c)
        for c in cols:
            self.y_list.addItem(_checkable_item(c))
            self.r_list.addItem(_checkable_item(c))
        # Defaults: X = the scanned positioner, Y = an intensity-like column.
        if cols:
            x_default, y_default = self._guess_axes(cols)
            if x_default:
                self.x_combo.setCurrentText(x_default)
            if y_default is not None:
                items = self.y_list.findItems(y_default, QtCore.Qt.MatchFlag.MatchExactly)
                if items:
                    items[0].setCheckState(QtCore.Qt.CheckState.Checked)
        self.x_combo.blockSignals(False)
        self.norm_combo.blockSignals(False)
        self.y_list.blockSignals(False)
        self.r_list.blockSignals(False)

        self.save_btn.setEnabled(bool(self._table))   # CSV exports all columns
        if not self._table:
            self.status.setText("No metadata found for this source.")
        elif not cols:
            self.status.setText(
                f"{len(self._table)} column(s), none numerically plottable.")
        else:
            self.status.setText(
                f"{len(cols)} plottable column(s). Pick X and check Y column(s).")
        self._redraw()

    #: Default-Y counter preference (first present wins; matched
    #: case-insensitively).  ROI columns are deliberately absent — neither a
    #: beamline ROI counter nor a computed ROI is ever auto-selected as the
    #: default (a frame-0 ROI says nothing about the scan).
    _Y_PRIORITY = ("Photod", "bs", "mon", "i2", "i1", "i0")

    def _guess_axes(self, cols):
        """(x_default, y_default).

        X = the scanned NeXus positioner (the recorded scanned motor) when one
        is present and actually varies, else ``frame_index``.  Y = the first
        present column of :attr:`_Y_PRIORITY` (case-insensitive), never an ROI
        column, else ``frame_index``."""
        x = self._scanned_positioner(cols)
        if x is None:
            x = "frame_index" if "frame_index" in cols else (cols[0] if cols else None)
        y = self._priority_y(cols, exclude=x)
        return x, y

    def _scanned_positioner(self, cols):
        """The X default: among the file's positioners present as plottable
        columns, the one that varies the most (a constant positioner wasn't the
        scan axis).  ``None`` when none are present or vary."""
        best, spread = None, 0.0
        for name in self._positioner_names:
            if name not in cols:
                continue
            a = np.asarray(self._table[name], dtype=float)
            a = a[np.isfinite(a)]
            if a.size < 2:
                continue
            s = float(np.nanmax(a) - np.nanmin(a))
            if s > spread:
                spread, best = s, name
        return best

    def _priority_y(self, cols, *, exclude):
        """The Y default: first present of :attr:`_Y_PRIORITY` (case-insensitive
        column match), skipping ``exclude``; never an ROI column.  Falls back to
        ``frame_index`` then ``None``."""
        by_lower = {c.lower(): c for c in cols}
        for name in self._Y_PRIORITY:
            col = by_lower.get(name.lower())
            if col is not None and col != exclude:
                return col
        if "frame_index" in cols and exclude != "frame_index":
            return "frame_index"
        return None

    @staticmethod
    def _checked_in(list_widget):
        out = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _checked_y(self):
        return self._checked_in(self.y_list)

    def _right_cols(self):
        """Columns assigned to the right-hand axis.  Checking a column here
        plots it on the right axis even if it isn't checked in Y, so the
        right-axis toggle is never a silent no-op."""
        return set(self._checked_in(self.r_list))

    def _redraw(self):
        self.plot.clear()
        self.right_vb.clear()
        if self.legend is not None:
            self.legend.clear()
        if not self._columns:
            self.right_axis.setVisible(False)
            return
        xcol = self.x_combo.currentText()
        if xcol not in self._table:
            self.right_axis.setVisible(False)
            return
        x = np.asarray(self._table[xcol], dtype=float)
        norm_col = self.norm_combo.currentData()
        norm = None
        if norm_col and norm_col in self._table:
            norm = np.asarray(self._table[norm_col], dtype=float)
            norm = np.where(norm == 0, np.nan, norm)   # avoid /0
        normalized = norm is not None
        right_cols = self._right_cols()
        # A column plots if it's checked in Y OR assigned to the right axis;
        # draw in the stable column order for deterministic colours/legend.
        y_cols = set(self._checked_y())
        draw_cols = [c for c in self._columns if c in y_cols or c in right_cols]
        any_left = any_right = False
        for n, ycol in enumerate(draw_cols):
            y = np.asarray(self._table[ycol], dtype=float)
            if normalized:
                y = y / norm
            if not (np.isfinite(x).any() and np.isfinite(y).any()):
                continue
            color = CURVE_PENS[n % len(CURVE_PENS)]
            pen = pg.mkPen(color, width=3)
            if ycol in right_cols:
                add_right_series(self.right_vb, self.legend, x, y, pen=pen,
                                 name=ycol, symbol="o", symbol_size=7,
                                 symbol_brush=color)
                any_right = True
            else:
                self.plot.plot(x, y, pen=pen, name=ycol, symbol="o",
                               symbolSize=7, symbolBrush=color)
                any_left = True
        self.plot.setLabel("bottom", xcol)
        value_label = ("value / " + norm_col) if normalized else "value"
        self.plot.setLabel("left", value_label if any_left else "")
        self.right_axis.setVisible(any_right)
        self.right_vb.setVisible(any_right)
        if any_right:
            self.right_axis.setLabel(f"{value_label} (right)")
        # Re-apply the Y-log toggle: plot.clear() at the top of every _redraw
        # resets per-axis log mode, so the state lives on the button.  Log the
        # LEFT axis only, then force the right axis back to linear: its curves
        # live in a separate ViewBox that PlotItem.setLogMode does NOT transform,
        # so without this the right ticks would go log while the data stayed
        # linear (mislabelled).  Right axis = linear ticks + linear data.
        self.plot.setLogMode(False, self.log_btn.isChecked())
        self.right_axis.setLogMode(False)

    def _schedule_redraw(self):
        if self._closing:
            return
        if not self._roi_redraw_timer.isActive():
            self._roi_redraw_timer.start()

    # ---- ROI columns ----------------------------------------------------
    def _update_roi_button(self):
        self.roi_btn.setEnabled(bool(self._raw_reachable))
        if self._raw_reachable:
            self.roi_btn.setToolTip(
                "Reduce rectangular ROIs over the scan's raw frames and add each "
                "as a plotted column")
        elif self._source is None:
            self.roi_btn.setToolTip(
                "This scan exposes no raw frames (metadata only) — ROI plotting "
                "is unavailable")
        else:
            self.roi_btn.setToolTip(
                "The scan's raw frames aren't reachable on disk — ROI plotting is "
                "unavailable")

    def _table_len(self):
        return max((len(np.atleast_1d(v)) for v in self._table.values()), default=0)

    def _frame_row_map(self):
        """``{frame_index: table row}`` for aligning streamed ROI stats to the
        table; positional when there is no ``frame_index`` column."""
        fi = self._table.get("frame_index")
        if fi is None:
            return {i: i for i in range(self._table_len())}
        out = {}
        for r, v in enumerate(np.asarray(fi, dtype=float)):
            if np.isfinite(v):
                out[int(round(v))] = r
        return out

    def _unique_col_name(self, base):
        base = base or "roi"
        if base not in self._table:
            return base
        i = 2
        while f"{base}_{i}" in self._table:
            i += 1
        return f"{base}_{i}"

    def _append_column(self, name, arr, *, check=False):
        """Add a per-frame column to the table + the X/Norm/Y selectors (so an
        ROI stat plots/overlays/normalizes/exports exactly like a metadata
        column).  Signals are blocked so adding items never spuriously redraws."""
        self._table[name] = np.asarray(arr, dtype=float)
        if name in self._columns:
            return
        self._columns.append(name)
        self.x_combo.blockSignals(True)
        self.norm_combo.blockSignals(True)
        self.y_list.blockSignals(True)
        self.r_list.blockSignals(True)
        try:
            self.x_combo.addItem(name)
            self.norm_combo.addItem(name, name)
            item = _checkable_item(name)
            if check:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.y_list.addItem(item)
            self.r_list.addItem(_checkable_item(name))
        finally:
            self.x_combo.blockSignals(False)
            self.norm_combo.blockSignals(False)
            self.y_list.blockSignals(False)
            self.r_list.blockSignals(False)
        self.save_btn.setEnabled(True)

    def _remove_column(self, name):
        self._table.pop(name, None)
        if name in self._columns:
            self._columns.remove(name)
        self.x_combo.blockSignals(True)
        self.norm_combo.blockSignals(True)
        self.y_list.blockSignals(True)
        self.r_list.blockSignals(True)
        try:
            i = self.x_combo.findText(name)
            if i >= 0:
                self.x_combo.removeItem(i)
            j = self.norm_combo.findText(name)
            if j >= 0:
                self.norm_combo.removeItem(j)
            for lst in (self.y_list, self.r_list):
                for k in range(lst.count() - 1, -1, -1):
                    if lst.item(k).text() == name:
                        lst.takeItem(k)
        finally:
            self.x_combo.blockSignals(False)
            self.norm_combo.blockSignals(False)
            self.y_list.blockSignals(False)
            self.r_list.blockSignals(False)

    def _open_roi_dialog(self):
        if not self._raw_reachable or self._source is None:
            self.status.setText(
                "Raw frames aren't reachable for this scan — ROI plotting is "
                "unavailable.")
            return
        image = self._first_image       # reuse the frame the reachability probe decoded
        if image is None:
            try:
                idxs = list(self._source.frame_indices)
                image = np.asarray(self._source.load_frame(idxs[0]))
            except Exception:
                logger.exception("scan-plot: could not load the first frame for ROI")
                self.status.setText("Could not load the first frame (see log).")
                return
        from .roi_select_dialog import RoiSelectDialog
        if self._roi_dialog is not None:
            self._roi_dialog.close()
        self._roi_dialog = RoiSelectDialog(image, parent=self)
        self._roi_dialog.sigCompute.connect(self._compute_roi)
        self._roi_dialog.show()
        self._roi_dialog.raise_()
        self._roi_dialog.activateWindow()

    def _compute_roi(self, signals):
        """Reduce ``signals`` (RoiSignals from the picker) over the raw frames
        off-thread; each becomes a NaN-seeded column that fills incrementally."""
        if self._source is None or not signals:
            return
        if self._roi_worker is not None and self._roi_worker.isRunning():
            self.status.setText("An ROI computation is already running.")
            return
        if "frame_index" not in self._table:
            try:
                idxs = [float(i) for i in self._source.frame_indices]
                self._append_column("frame_index", np.asarray(idxs))
            except Exception:
                pass
        self._roi_row_of = self._frame_row_map()
        n = self._table_len()
        named = []
        self._roi_run_columns = []
        for sig in signals:
            col = self._unique_col_name(sig.name or sig.roi.name or "roi")
            named.append(replace(sig, name=col))
            self._append_column(col, np.full(n, np.nan), check=True)
            self._roi_run_columns.append(col)
        from .analysis_worker import RoiStatsWorker
        if self._roi_worker is None:
            self._roi_worker = RoiStatsWorker(self)
            self._roi_worker.sigProgress.connect(self._on_roi_progress)
            self._roi_worker.sigFrameStat.connect(self._on_roi_frame)
            self._roi_worker.sigRoiDone.connect(self._on_roi_done)
        # Apply the same static detector mask the reducer uses (when the source
        # is the loaded scan) + the picker's saturation toggle, so ROI stats
        # agree with the reduction.
        mask = None
        if self._mask_provider is not None:
            try:
                mask = self._mask_provider(self._source_uri)
            except Exception:
                logger.exception("scan-plot: mask provider failed")
                mask = None
        mask_saturation = bool(self._roi_dialog is not None
                               and self._roi_dialog.mask_saturated())
        self._roi_worker.configure(named, self._source, x_key=None, mask=mask,
                                   mask_saturation=mask_saturation)
        self.status.setText("Computing ROI stats…")
        self._roi_worker.start()
        self._schedule_redraw()

    def _on_roi_progress(self, done, total):
        if self._closing:
            return
        self.status.setText(f"Computing ROI stats… {done}/{total}")

    def _on_roi_frame(self, frame_index, row):
        if self._closing:
            return
        r = self._roi_row_of.get(int(frame_index))
        if r is None:
            return
        for name, val in row.items():
            arr = self._table.get(name)
            if arr is not None and 0 <= r < len(arr):
                arr[r] = val
        self._schedule_redraw()

    def _on_roi_done(self, result):
        # The cancel/fail cleanup runs even while closing: a close mid-run stops
        # the worker, which emits sigRoiDone(None) AFTER _closing is set — so
        # dropping the partial columns must NOT sit behind the _closing guard, or
        # the orphan NaN columns survive into the next time the (reused) dialog
        # is shown.
        if result is None:                  # cancelled / failed -> drop partials
            self._roi_redraw_timer.stop()
            had = bool(self._roi_run_columns)
            for name in self._roi_run_columns:
                self._remove_column(name)
            self._roi_run_columns = []
            if had and not self._closing:    # quiet if a source-swap already cleared
                self.status.setText("ROI computation cancelled.")
                self._redraw()
            return
        # A run that COMPLETED keeps its columns; if the dialog is closing, just
        # finalise the bookkeeping without touching the UI.
        n_added = len(self._roi_run_columns)
        self._roi_redraw_timer.stop()
        self._roi_run_columns = []
        if self._closing:
            return
        # Flush the final coalesced frame(s): the last _on_roi_frame scheduled a
        # debounced redraw, but sigRoiDone (this slot) cancels the pending timer
        # above before it fires — so the plot would otherwise stay one update
        # stale until some other event redraws.  _redraw is idempotent.
        self._redraw()
        diag = getattr(result.payload, "diagnostics", {}) or {}
        no_raw = diag.get("no_raw_frames") or []
        msg = f"ROI stats done — {n_added} column(s) added."
        if no_raw:
            msg += f" ({len(no_raw)} frame(s) had unreachable raw → NaN.)"
        self.status.setText(msg)

    def keyPressEvent(self, event):
        # Esc must not discard the assembled table / ROI setup (the default
        # QDialog Esc -> reject closes the popup).  Other keys pass through.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        self._closing = False
        super().showEvent(event)

    def closeEvent(self, event):
        self._closing = True
        if self._roi_worker is not None:
            self._roi_worker.stop()
        if hasattr(self, "_roi_redraw_timer"):
            self._roi_redraw_timer.stop()
        if self._roi_dialog is not None:
            self._roi_dialog.close()
            self._roi_dialog = None
        # The source widget's probe executor is a NON-daemon thread and Qt
        # never delivers closeEvent to child widgets — without this, a probe
        # in flight at close survives as a live thread that concurrent.futures
        # joins at interpreter exit (post-summary hang in tests; app-exit hang
        # live) and whose done-callback then signals into a destroyed widget.
        if hasattr(self, "source_widget"):
            self.source_widget.shutdown_probe_worker()
        super().closeEvent(event)

    def _csv_columns(self):
        """Every 1-D column of the assembled table — including non-numeric (e.g.
        string motor) and all-NaN columns the plot selectors omit, so the CSV is
        the full per-frame table the user sees + computed."""
        out = []
        for name, arr in self._table.items():
            a = np.atleast_1d(np.asarray(arr, dtype=object))
            if a.ndim == 1:
                out.append(name)
        return out

    def _write_csv(self, path):
        """Write the assembled table to ``path`` (one column per key, blanks for
        ragged rows).  Separated from the file dialog so it's unit-testable."""
        import csv
        cols = self._csv_columns()
        arrays = {c: np.atleast_1d(self._table[c]) for c in cols}
        n = max((len(a) for a in arrays.values()), default=0)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for r in range(n):
                writer.writerow([arrays[c][r] if r < len(arrays[c]) else ""
                                 for c in cols])

    def _save_csv(self):
        if not self._table:
            return
        from xdart.utils.browse import remember_browse_path, suggest_save_path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save scan table", suggest_save_path("scan_table.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        remember_browse_path(path)
        try:
            self._write_csv(path)
            self.status.setText(f"Saved {path}")
        except Exception:
            logger.exception("scan-table CSV save failed")
            self.status.setText("Could not save CSV (see log).")
