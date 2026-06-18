# -*- coding: utf-8 -*-
"""Plot rendering, waterfall, slice overlay, and mouse-tracking methods
for displayFrameWidget.

This mixin extracts ~600 lines of 1D-plot and waterfall logic from the
monolithic displayFrameWidget class.

The mixin is designed to be inherited by displayFrameWidget alongside
QWidget and DisplayDataMixin, so all ``self`` references resolve to
the composite widget.
"""

import logging
import re

import numpy as np
import pyqtgraph as pg
from pyqtgraph import Qt, ROI
from pyqtgraph.Qt import QtWidgets

from .display_constants import (
    AA_inv, Th, Chi, Deg,
    x_labels_1D, x_units_1D,
)
from .display_logic import (
    plan_overlay,
    overlay_read_failure_action,
    OverlayAction,
    pretty_unit,
    nanmean_slice,
    resample_image_axis_to_uniform,
)

logger = logging.getLogger(__name__)


def _as_plot_rows(ydata):
    ydata = np.asarray(ydata, dtype=float)
    if ydata.ndim == 1:
        ydata = ydata[np.newaxis, :]
    return ydata


def _reinterp_plot_row(src_x, src_y, dst_x):
    """Interpolate one plot row onto ``dst_x``, NaN outside source range."""
    if np.size(src_x) == 0 or np.size(dst_x) == 0:
        return np.full(np.shape(dst_x), np.nan)
    out = np.interp(dst_x, src_x, src_y)
    out[dst_x < src_x[0]] = np.nan
    out[dst_x > src_x[-1]] = np.nan
    return out


def update_plot_accumulator(
    prev_plot_data,
    prev_names,
    prev_ids,
    new_x,
    new_y,
    new_names,
    new_ids,
    method,
    unit_changed,
):
    """Pure Overlay/Waterfall plot accumulator transform.

    The widget owns all persistent state and passes it in.  This function only
    computes the next ``plot_data`` / ``frame_names`` / ``overlaid_idxs`` tuple.
    """
    new_x = np.asarray(new_x, dtype=float)
    new_y = _as_plot_rows(new_y)
    names = list(prev_names or [])
    ids = [int(i) for i in (prev_ids or [])]

    overlay_action, _ = plan_overlay(
        method, unit_changed,
        has_existing=len(ids) > 0,
        new_ids=tuple(int(i) for i in new_ids),
        prev_overlaid_ids=tuple(ids),
    )

    if overlay_action is OverlayAction.REBUILD:
        # Monotonicity invariant: a REBUILD RE-EXPRESSES the accumulated frames in
        # a new unit -- it must reproduce the EXACT prior id set, never change it.
        # A REBUILD whose incoming ids differ from the accumulator is a CORRUPT
        # re-read: a spurious unit_changed sent us down the re-read fallback, and
        # the cap-store had evicted older frames, so the non-blocking re-read came
        # back partial/holey and prefix-MISLABELED (e.g. resident frames 37..100
        # shown as scan_1..scan_64).  Reject it and KEEP the prior accumulator
        # rather than shrink+relabel the visible stack (the intermittent
        # collapse-and-restack regression).  Legitimate resets (Clear / new scan /
        # mode switch) go through clear_overlay -> prev ids empty -> plan_overlay
        # returns REPLACE (not REBUILD), so they never reach this guard.  A genuine
        # same-unit-count Q<->2theta re-express reproduces the id set and passes.
        if set(int(i) for i in new_ids) != set(ids):
            return list(prev_plot_data), list(names), list(ids)
        return [new_x, new_y], list(new_names), [int(i) for i in new_ids]

    if overlay_action is not OverlayAction.APPEND:
        return [new_x, new_y], list(new_names), [int(i) for i in new_ids]

    prev_x, prev_y = prev_plot_data
    prev_x = np.asarray(prev_x, dtype=float)
    prev_y = _as_plot_rows(prev_y)
    plot_data = [prev_x, prev_y]

    for idx, frame_name, row in zip(new_ids, new_names, new_y):
        if frame_name in names:
            continue
        # Skip an empty-grid incoming frame: it carries no usable x axis and
        # would poison the accumulator.
        if np.size(new_x) == 0:
            continue
        old_x = np.asarray(plot_data[0], dtype=float)
        if np.size(old_x) == 0:
            # An empty x grid means the visible accumulator was cleared or never
            # initialized.  Start a fresh, internally consistent history instead
            # of carrying stale ids/names forward.
            plot_data = [new_x, row[np.newaxis, :]]
            names = []
            ids = []
        elif old_x.shape == new_x.shape and np.allclose(old_x, new_x):
            old_y = _as_plot_rows(plot_data[1])
            plot_data[1] = np.vstack((old_y, row))
        else:
            merged_x = np.union1d(old_x, new_x)
            merged_x.sort()
            old_y = _as_plot_rows(plot_data[1])
            new_old = np.array([
                _reinterp_plot_row(old_x, r, merged_x)
                for r in old_y
            ])
            new_row = _reinterp_plot_row(new_x, row, merged_x)
            plot_data = [merged_x, np.vstack((new_old, new_row))]
        names.append(frame_name)
        ids.append(int(idx))

    return plot_data, names, ids


class DisplayPlotMixin:
    """Mixin providing 1D plot, waterfall, slice overlay, and mouse helpers.

    Expects the host widget to expose at least:

    - ``self.ui`` (the Ui_Form instance)
    - ``self.scan``, ``self.frame_ids``, ``self.data_1d``, ``self.data_2d``
    - ``self.idxs``, ``self.idxs_1d``
    - ``self.plot``, ``self.plot_win``, ``self.plot_layout``
    - ``self.wf_widget``, ``self.wf_*`` (waterfall state)
    - ``self.curves``, ``self.legend``, ``self.pos_label``
    - ``self.plot_data``, ``self.plot_data_range``, ``self.frame_names``
    - ``self.plotMethod``, ``self.scale``, ``self.cmap``
    - ``self.overlay``, ``self.binned_data``, ``self.binned_widget``
    - ``self._plot_axis_info``, ``self._last_plot_unit``
    - Methods from DisplayDataMixin: ``get_frames_int_1d``, ``get_colors``,
      ``normalize``, ``show_slice_overlay``
    """

    # ── 1D plot data accumulation ─────────────────────────────────

    def resolve_plot_axis(self):
        """Return the selected plot-axis metadata and whether it needs 2D."""
        _idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[_idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= _idx < len(self._plot_axis_info)
                else None)
        needs_2d = ((info and info['source'] in ('2d', '1d_2d'))
                    or self.ui.slice.isChecked())
        return info, needs_2d

    def collect_plot_rows(self):
        """Collect x/y rows for the current plot selection.

        Preserves the legacy fallback: when the selected axis needs 2D data
        but only 1D data is available, disable slicing and retry on plain 1D.
        """
        method = self.ui.plotMethod.currentText()
        accumulating = method in ("Overlay", "Waterfall")
        idle = not getattr(self, "_processing_active", False)
        if accumulating:
            # Overlay/Waterfall accumulate into plot_data / overlaid_idxs.  Fetch
            # ONLY the frames not already accumulated -- NEVER block-read the whole
            # scan's 1D from disk on the GUI thread.  That was the end-of-scan
            # HARD FREEZE: the reduction finishes ahead of the display, the mode
            # flips live->idle, and at catch-up (e.g. 648 of 651 rows already
            # accumulated) the old code re-read ALL 651 frames from disk with
            # require_all -> a multi-second GUI-thread read that, when the last
            # frames aren't all readable yet, returns None and is retried every
            # tick (autorange / aggregate worker / flush) -> never recovers.
            # A fresh RELOAD has an empty accumulator, so it still reads the full
            # selection (once).  Block from disk only when idle; during live, skip
            # evicted frames -- they arrive via the next append.  require_all is
            # False so a partial read appends what it can instead of blanking.
            selection = [int(i) for i in (self.idxs if idle else self.idxs_1d)]
            have = {int(i) for i in (getattr(self, "overlaid_idxs", []) or [])}
            missing = [i for i in selection if i not in have]
            idxs = missing if (have and missing) else selection
            require_all = False
            # NEVER block-read on the GUI thread for an accumulating display.
            # allow_blocking_read=None routes through the async path: in the live
            # app (async hydration on) this reads only resident rows now and the
            # FrameHydrationWorker backfills the rest off-thread (no freeze on a
            # unit switch / reload / catch-up where the whole accumulator is the
            # selection); headless tests (async off) still block-read so their
            # first render has the data.
            allow_blocking = None
        else:
            store_complete_required = (
                getattr(self, "viewer_mode", None) is None
                and getattr(self, "publication_store", None) is not None
                and method in ("Single", "Sum", "Average")
            )
            idxs = self.idxs if store_complete_required else None
            require_all = store_complete_required
            allow_blocking = None
        self._plot_row_ids = list(self.idxs_1d if idxs is None else idxs)
        ydata, xdata = self.get_frames_int_1d(
            idxs,
            require_all=require_all,
            allow_blocking_read=allow_blocking,
        )
        if xdata is not None and ydata is not None:
            return ydata, xdata

        _info, needs_2d = self.resolve_plot_axis()
        if needs_2d and getattr(self.scan, 'skip_2d', False):
            try:
                self.window().statusBar().showMessage(
                    "Chi slicing requires 2D integration (1D Only is enabled).", 4000)
            except Exception:
                logger.debug("Failed to show status bar message about chi slicing", exc_info=True)

        # Fall back: disable slice, retry with plain 1D.
        self.ui.slice.setChecked(False)
        return self.get_frames_int_1d(
            idxs,
            require_all=require_all,
            allow_blocking_read=allow_blocking,
        )

    def build_plot_names(self, idxs=None):
        """Return current trace names, including slice-range suffixes."""
        if idxs is None:
            idxs = self.idxs
        idxs = list(idxs)
        if self.scan.series_average:
            frame_names = [self.scan.name]
        else:
            frame_names = [f'{self.scan.name}_{i}' for i in idxs]

        # When slicing is active, include slice parameters in frame names
        # so the same image with different slice ranges can be overlaid.
        if self.ui.slice.isEnabled() and self.ui.slice.isChecked():
            center = self.ui.slice_center.value()
            width = self.ui.slice_width.value()
            suffix = f' [{center:.1f}\u00b1{width:.1f}]'
            frame_names = [n + suffix for n in frame_names]
        return frame_names

    def apply_plot_background(self, ydata):
        """Subtract the current 1D background and return row-shaped ydata."""
        if self.bkg_1d is not None:
            ydata -= self.bkg_1d
        if ydata.ndim == 1:
            ydata = ydata[np.newaxis, :]
        return ydata

    def _reexpress_overlay_unit(self, prev_unit_idx, new_unit_idx):
        """Re-express the accumulated Overlay/Waterfall x-axis in the new plotUnit
        WITHOUT re-reading from disk, for a pure 1D radial display-unit toggle
        (Q<->2θ): the accumulated intensity is unit-invariant, so only the shared
        x-axis transforms.

        Returns True on success (``self.plot_data[0]`` replaced; the y rows,
        ``frame_names`` and ``overlaid_idxs`` are untouched), False on any guard
        miss — the caller then falls back to the non-blocking re-read.  This is
        what makes a Q<->2θ switch instant + COMPLETE: no re-read happens, so there
        is no cap-store backfill race (the cause of the half-filled waterfall).

        Reuses :meth:`get_xdata` (the single blessed Q<->2θ + wavelength path) via a
        stub frame that carries the existing grid in its OLD unit, so the
        conversion matches the per-frame readout exactly — no formula drift."""
        try:
            from types import SimpleNamespace
            from .display_constants import Th
            pd = getattr(self, "plot_data", None)
            if (not pd or len(pd) != 2
                    or np.size(pd[0]) == 0 or np.size(pd[1]) == 0):
                return False
            old_x = np.asarray(pd[0], dtype=float)
            y = _as_plot_rows(pd[1])
            if y.ndim != 2 or old_x.shape[0] != y.shape[1]:
                return False                  # single-shared-grid invariant broken
            info = getattr(self, "_plot_axis_info", None)
            try:
                sliced = bool(self.ui.slice.isChecked())
            except Exception:
                sliced = False

            def pure_radial(idx):
                if not info or not (0 <= idx < len(info)):
                    return False
                src = (info[idx] or {}).get("source")
                # χ is source '2d'; a sliced 1d_2d axis projects from the cake --
                # neither is a unit-invariant 1D readout.
                return not ((src == '2d') or (src == '1d_2d' and sliced))

            if not (pure_radial(prev_unit_idx) and pure_radial(new_unit_idx)):
                return False
            old_label = self.ui.plotUnit.itemText(prev_unit_idx)
            new_label = self.ui.plotUnit.currentText()
            old_unit = '2th_deg' if Th in old_label else 'q_A^-1'
            stub = SimpleNamespace(
                int_1d=SimpleNamespace(radial=old_x, unit=old_unit))
            new_x = self.get_xdata(stub)
            if new_x is None:
                return False
            new_x = np.asarray(new_x, dtype=float)
            if new_x.shape[0] != y.shape[1] or not np.all(np.isfinite(new_x)):
                return False
            # Family changed but get_xdata returned the grid unconverted (no
            # wavelength) -> would mislabel the axis; defer to the re-read path.
            if (Th in old_label) != (Th in new_label) and np.allclose(new_x, old_x):
                return False
            self.plot_data = [new_x, pd[1]]
            return True
        except Exception:
            logger.debug("overlay unit re-express failed; falling back to re-read",
                         exc_info=True)
            return False

    def compute_plot_range(self, xdata, ydata):
        """Update ``plot_data_range`` for non-empty plot arrays."""
        if xdata.size == 0 or ydata.size == 0:
            return False
        self.plot_data_range = [
            [np.nanmin(xdata), np.nanmax(xdata)],
            [np.nanmin(ydata), np.nanmax(ydata)],
        ]
        return True

    def draw_plot_state(self):
        """Draw the currently accumulated plot state."""
        self.update_plot_view()
        if getattr(self, '_plot_autorange_requested', False):
            self._plot_autorange_requested = False
            self._autorange_plot_view()

    def request_plot_autorange(self, *args):
        """Request a 1D autorange after the next canonical plot draw."""
        self._plot_autorange_requested = True

    def _loaded_1d_overlay_labels(self, idxs, *, max_rows=None):
        """Return loaded frame ids and labels that can produce 1D rows."""
        kept = []
        names = []
        slice_suffix = ''
        try:
            if self.ui.slice.isEnabled() and self.ui.slice.isChecked():
                center = self.ui.slice_center.value()
                width = self.ui.slice_width.value()
                slice_suffix = f' [{center:.1f}\u00b1{width:.1f}]'
        except Exception:
            slice_suffix = ''
        if hasattr(self, "_snapshot_data"):
            snapshot = self._snapshot_data(idxs)
        else:
            data_1d = getattr(self, "data_1d", {}) or {}
            data_2d = getattr(self, "data_2d", {}) or {}
            snapshot = {
                int(idx): (data_1d.get(int(idx)), data_2d.get(int(idx), {}))
                for idx in idxs
            }
        for idx in idxs:
            idx = int(idx)
            frame_1d, frame_2d = snapshot.get(idx, (None, None))
            if frame_1d is None:
                continue
            try:
                x, y = self.get_int_1d(frame_1d, frame_2d, idx)
            except Exception:
                logger.debug("overlay label rebuild skipped frame %s",
                             idx, exc_info=True)
                continue
            if x is None or y is None:
                continue
            kept.append(idx)
            names.append(f'{self.scan.name}_{idx}{slice_suffix}')
            if max_rows is not None and len(kept) >= max_rows:
                break
        return kept, names

    def update_plot(self):
        """Updates data in plot frame
        """
        # Legacy path owns its axis label from the plotUnit combo.  After the
        # Step-5 flip a payload render may have stashed _payload_x/y_axis_label
        # (read first by _current_plot_axis_label / update_1d_view); clear it so
        # a control-driven legacy redraw (plotUnit/slice still wired to
        # update_plot) doesn't show the stale payload label.
        self._payload_x_axis_label = None
        self._payload_y_axis_label = None
        if (self.scan.name == 'null_main') or (len(self.frame_ids) == 0):
            data = (np.arange(100), np.arange(100))
            return data

        # FREEZE FIX (live Overlay/Waterfall, end-of-scan): when the accumulator
        # already covers every selected frame in the CURRENT unit, just redraw —
        # do NOT re-collect.  Otherwise an idle re-render (the scan ends ->
        # _processing_active=False -> collect_plot_rows takes the blocking branch)
        # block-reads the WHOLE scan's 1D from disk on the GUI thread (~1 s for a
        # 651-frame scan whose heavy rows were evicted past the store cap) on
        # EVERY refresh — repeated by autorange / the aggregate worker / the final
        # flush, it saturates the GUI thread into a hard freeze.  The rows are
        # already in plot_data from live accumulation, so a reload (empty
        # accumulator) still falls through to read from disk once.
        method = self.ui.plotMethod.currentText()
        # NEW-SCAN BOUNDARY RESET: the Overlay/Waterfall accumulator is stamped
        # with the scan it was built from.  When the current scan differs, drop it
        # so the new scan plots FRESH -- a new scan may use different integration
        # params / GI / a different axis, so appending across scans would mix
        # incompatible data (Vivek-confirmed: reset, don't append).  Self-heals any
        # scan-swap path that didn't route through new_scan's clear_overlay, and is
        # consistent for <15 (curves) and >15 (waterfall).  Same-scan renders
        # (incl. the in-scan end-of-scan catch-up skip below) are untouched.
        _scan_key = getattr(self.scan, "data_file", None) or getattr(
            self.scan, "name", None)
        if (method in ("Overlay", "Waterfall")
                and getattr(self, "overlaid_idxs", None)
                and getattr(self, "_overlay_scan_key", None) not in (None, _scan_key)):
            # New-scan boundary: drop the stale accumulator.  Inline (clear_overlay
            # is a displayFrameWidget method; update_plot is a shared mixin method
            # also exercised by duck-host tests).  plot_data_range is recomputed by
            # compute_plot_range on the next draw.
            self.plot_data = [np.zeros(0), np.zeros(0)]
            self.frame_names = []
            self.overlaid_idxs = []
        self._overlay_scan_key = _scan_key

        if (method in ("Overlay", "Waterfall")
                and getattr(self, "overlaid_idxs", None)
                and self.ui.plotUnit.currentIndex()
                == getattr(self, "_last_plot_unit", -1)):
            sel = {int(i) for i in self.idxs_1d}
            if sel and sel <= {int(i) for i in self.overlaid_idxs}:
                self.draw_plot_state()
                return

        # Get 1D data for all frames
        ydata, xdata = self.collect_plot_rows()
        if xdata is None or ydata is None:
            # Append-only invariant: a failed/partial INCREMENTAL read must never
            # wipe an existing Overlay/Waterfall accumulator -- the missing frames
            # are in flight (being written, or evicted past the store cap and
            # awaiting async hydration) and arrive on a later tick.  Clearing here
            # was the regression where a slow GUI (e.g. Share Axis) let the
            # reduction race ahead, the non-blocking read of the newest frames
            # returned None, and the whole stack collapsed to ~0 and re-stacked as
            # it caught up.  PRESERVE keeps the accumulator and redraws it.
            if overlay_read_failure_action(
                method, bool(getattr(self, "overlaid_idxs", None))
            ) == "preserve":
                self.draw_plot_state()
            else:
                self.clear_plot_view()
            return

        row_ids = list(getattr(self, "_plot_row_ids", self.idxs_1d))
        row_count = _as_plot_rows(ydata).shape[0]
        row_ids = row_ids[:row_count]
        frame_names = self.build_plot_names(row_ids)

        # Subtract background
        ydata = self.apply_plot_background(ydata)

        # Single-mode overplots whatever rows are in the current
        # selection: a single-click gives one curve, shift/cmd-click
        # gives several.  Live-scan selection accumulation (which used
        # to silently turn Single into a Waterfall) is now prevented at
        # the source by ClearAndSelect in h5viewer + latest_frame, so we
        # don't need to narrow ydata here.

        current_plot_unit = self.ui.plotUnit.currentIndex()
        prev_plot_unit = self._last_plot_unit
        unit_changed = current_plot_unit != prev_plot_unit
        self._last_plot_unit = current_plot_unit

        # In Overlay/Waterfall: accumulate new frames, skip duplicates.
        # On a unit change, rebuild the accumulated set in the new unit
        # instead of appending across incompatible x grids or dropping all
        # prior curves.  In Single/Sum/Average: always replace with the
        # current selection.  Stage 4: the accumulate/rebuild/replace choice
        # is the pure plan_overlay (unit-tested headlessly); the branch
        # bodies below still own the array work + eviction filtering.
        current_method = self.ui.plotMethod.currentText()
        overlay_action, _ = plan_overlay(
            current_method, unit_changed,
            has_existing=len(self.overlaid_idxs) > 0,
            new_ids=tuple(self.idxs_1d),
            prev_overlaid_ids=tuple(self.overlaid_idxs),
        )

        if overlay_action is OverlayAction.REBUILD:
            if self._reexpress_overlay_unit(prev_plot_unit, current_plot_unit):
                # Pure Q<->2θ display-unit toggle: the accumulated intensities are
                # unit-invariant, so re-express ONLY the shared x-axis in place --
                # NO disk re-read at all.  This is what fixes the unit-switch
                # incompleteness: the prior non-blocking re-read raced the cap-64
                # store (frames hydrated then evicted before being appended), so
                # the waterfall came back half-filled.  y / frame_names /
                # overlaid_idxs are untouched.
                pass
            else:
                rebuild_idxs = list(self.overlaid_idxs)
                # Fallback (non-pure-radial change, no resident frame, missing
                # wavelength): re-express by re-reading, but WITHOUT block-reading
                # the whole scan on the GUI thread (the unit-switch freeze).
                # allow_blocking_read=None routes through the async path -- with
                # the live app's async hydration on (enabled at startup) this reads
                # resident rows now and the FrameHydrationWorker backfills the rest
                # off-thread.  Headless tests (async OFF) still block-read for
                # determinism.  require_all=False so a partial read is applied.
                y_new, x_new = self.get_frames_int_1d(
                    rebuild_idxs,
                    require_all=False,
                    allow_blocking_read=None,
                )
                if x_new is not None and y_new is not None:
                    y_new = self.apply_plot_background(y_new)
                    kept = rebuild_idxs[:_as_plot_rows(y_new).shape[0]]
                    kept_names = self.build_plot_names(kept)
                    self.plot_data, self.frame_names, self.overlaid_idxs = (
                        update_plot_accumulator(
                            self.plot_data,
                            self.frame_names,
                            self.overlaid_idxs,
                            x_new,
                            y_new,
                            kept_names,
                            kept,
                            current_method,
                            unit_changed,
                        )
                    )
                else:
                    self.plot_data, self.frame_names, self.overlaid_idxs = (
                        update_plot_accumulator(
                            self.plot_data,
                            self.frame_names,
                            self.overlaid_idxs,
                            xdata,
                            ydata,
                            frame_names,
                            row_ids,
                            current_method,
                            unit_changed,
                        )
                    )
        elif overlay_action is OverlayAction.APPEND:
            self.plot_data, self.frame_names, self.overlaid_idxs = (
                update_plot_accumulator(
                    self.plot_data,
                    self.frame_names,
                    self.overlaid_idxs,
                    xdata,
                    ydata,
                    frame_names,
                    row_ids,
                    current_method,
                    unit_changed,
                )
            )
        else:
            # Fresh start: Single/Sum/Average, unit changed, or no existing data
            self.plot_data, self.frame_names, self.overlaid_idxs = (
                update_plot_accumulator(
                    self.plot_data,
                    self.frame_names,
                    self.overlaid_idxs,
                    xdata,
                    ydata,
                    frame_names,
                    row_ids,
                    current_method,
                    unit_changed,
                )
            )

        xdata, ydata = self.plot_data
        if not self.compute_plot_range(xdata, ydata):
            return

        self.draw_plot_state()

    def _on_plotMethod_changed(self):
        """Handle plotMethod combo box changes.

        Emits ``sigPlotMethodChanged`` so external widgets (e.g. the
        H5Viewer's data list) can adapt their selection mode.

        Switching to Single mode resets accumulated plot data so the
        plot rebuilds from the current selection. Overlay/Waterfall and
        Sum/Average all rely on the user's full multi-selection in
        listData and route through update_plot() so the active set is
        re-aggregated each time.
        """
        new_method = self.ui.plotMethod.currentText()
        try:
            self.sigPlotMethodChanged.emit(new_method)
        except Exception:
            logger.debug("sigPlotMethodChanged emit failed", exc_info=True)

        if getattr(self, 'viewer_mode', None) == 'xye':
            if new_method in ('Single', 'Sum', 'Average'):
                self.clear_overlay()
            # Re-render through the payload path (XYEViewerController.build_payload);
            # Overlay/Waterfall accumulation lives there now, keyed off the
            # widget's plot_data/frame_names (which clear_overlay resets above).
            self.update()
            self._autorange_plot_view()
            return

        self.request_plot_autorange()
        if new_method == 'Single':
            # Reset accumulated data — rebuild from current selection
            self.plot_data = [np.array([]), np.array([])]
            self.frame_names = []
            self.overlaid_idxs = []
            if hasattr(self, "get_idxs"):
                self.get_idxs()
            self.update_plot()
        elif new_method in ('Sum', 'Average'):
            # No accumulation needed: aggregation happens inside
            # update_1d_view() based on the current selection.
            self.plot_data = [np.array([]), np.array([])]
            self.frame_names = []
            self.overlaid_idxs = []
            if hasattr(self, "get_idxs"):
                self.get_idxs()
            self.update_plot()
        else:
            # Overlay / Waterfall: keep existing accumulated curves and
            # just refresh the rendered view.
            self.draw_plot_state()

    def _current_plot_axis_label(self):
        """Return the bottom-axis label and unit for the current 1D view."""
        if getattr(self, 'viewer_mode', None) == 'xye':
            axis = getattr(self, '_viewer_x_axis_label', None)
            if axis is not None:
                return axis[0], pretty_unit(axis[1])
        axis = getattr(self, '_payload_x_axis_label', None)
        if axis is not None:
            return axis[0], pretty_unit(axis[1])

        plot_text = self.ui.plotUnit.currentText()
        m = re.match(r'^(.+?)\s*\((.+)\)$', plot_text)
        if m:
            return m.group(1).strip(), pretty_unit(m.group(2).strip())
        return plot_text, ''

    # ── 1D plot view rendering ────────────────────────────────────

    def update_plot_view(self):
        """Updates 1D view of data in plot frame
        """
        using_publication = getattr(self, '_using_publication_plot_payload', False)
        store = getattr(self, 'publication_store', None)
        has_store_rows = False
        if store is not None:
            try:
                has_store_rows = len(store) > 0
            except Exception:
                has_store_rows = False
        if (
            len(self.frame_ids) == 0
            or (
                len(self.data_1d) == 0
                and not using_publication
                and not has_store_rows
            )
        ):
            return

        # Clear curves.  removeItem, not curve.clear(): PlotDataItem.clear()
        # only blanks the data and leaves the item (plus its child curve +
        # scatter items) registered on the PlotItem/ViewBox -- they
        # accumulated per render, growing the scene and making autorange's
        # childrenBounds() sweep quadratically slower over a long run.
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves.clear()

        self.plotMethod = self.ui.plotMethod.currentText()

        self.ui.yOffset.setEnabled(False)
        # yOffset is only meaningful when more than one curve is on screen.
        # That's Overlay with a multi-selection, or Single mode with a
        # manual shift/cmd-click multi-selection (which behaves like Overlay).
        if (self.plotMethod in ('Overlay', 'Single')
                and len(self.frame_names) > 1):
            self.ui.yOffset.setEnabled(True)

        n_curves = len(self.plot_data[1])
        # Auto-switch to a waterfall only when MANY separate curves would be drawn
        # — Overlay/Waterfall, or a Single multi-selection.  Sum/Average COLLAPSE
        # to one curve in update_1d_view (nanmean/nansum over the stacked rows), so
        # they must NEVER auto-waterfall however many rows are stacked (the payload
        # path emits one un-reduced trace per frame, so n_curves is the frame
        # count, not the drawn-curve count).
        waterfall_active = getattr(self, '_waterfall_active', None)
        auto_wf = (
            waterfall_active()
            if callable(waterfall_active)
            else (
                (self.plotMethod == 'Waterfall' and n_curves > 3)
                or (self.plotMethod not in ('Sum', 'Average') and n_curves > 15)
            )
        )
        if auto_wf:
            self.update_wf()
        else:
            self.update_1d_view()

    def update_1d_view(self):
        """Updates data in 1D plot Frame
        """
        self.setup_1d_layout()

        xdata_, ydata_ = self.plot_data
        s_xdata, ydata = xdata_.copy(), ydata_.copy()

        # Nothing to draw (e.g. an all-evicted / all-empty GI selection): clear and
        # bail BEFORE the scale math + the Sum/Average nanmean/nansum, which would
        # otherwise warn "Mean of empty slice" on a zero-row stack.
        if np.size(ydata) == 0:
            self.setup_curves()
            return s_xdata, ydata

        int_label = 'I'
        if self.normChannel:
            int_label = f'I / {self.normChannel}'

        self.plot.getAxis("left").setLogMode(False)
        self.plot.getAxis("bottom").setLogMode(False)
        ylabel = f'{int_label} (a.u.)'
        payload_y_axis = getattr(self, '_payload_y_axis_label', None)
        if payload_y_axis is not None:
            y_label, y_unit = payload_y_axis
            ylabel = y_label
        if self.scale == 'Log':
            if ydata.size == 0:
                return
            if ydata.min() < 1:
                ydata -= (ydata.min() - 1.)
            ydata = np.log10(ydata)
            self.plot.getAxis("left").setLogMode(True)
            ylabel = f'Log {int_label}(a.u.)'
        elif self.scale == 'Sqrt':
            if ydata.min() < 0.:
                ydata_ = np.sqrt(np.abs(ydata))
                ydata_[ydata < 0] *= -1
                ydata = ydata_
            else:
                ydata = np.sqrt(ydata)
            ylabel = f'<math>&radic;</math>{int_label} (a.u.)'

        # Overlay/Waterfall always stack the selection.  Single also
        # stacks when the user has multi-selected frames (shift/cmd-click) —
        # that branch behaves like Overlay but without the yOffset.
        multi_single = self.plotMethod == 'Single' and ydata.shape[0] > 1
        if self.plotMethod in ['Overlay', 'Waterfall'] or multi_single:
            ydata = ydata[self.wf_start:self.wf_stop:self.wf_step]
            self.setup_curves()

            offset = self.ui.yOffset.value()
            y_offset = offset / 100 * (self.plot_data_range[1][1] - self.plot_data_range[1][0])
            for nn, (curve, s_ydata) in enumerate(zip(self.curves, ydata)):
                curve.setData(s_xdata, s_ydata + y_offset*nn)

        else:
            self.setup_curves()
            s_ydata = ydata
            if self.plotMethod == 'Average':
                # nanmean_slice suppresses "Mean of empty slice" when a column is
                # all-NaN across the selected frames (GI empty bins) — the NaN gap
                # is kept, just not warned.  (Sum's nansum doesn't warn.)
                reduced = nanmean_slice(s_ydata, 0)
                if reduced is not None:
                    s_ydata = reduced
            elif self.plotMethod == 'Sum':
                s_ydata = np.nansum(s_ydata, 0)

            self.curves[0].setData(s_xdata, s_ydata.squeeze())

        _xl, _xu = self._current_plot_axis_label()
        self.plot.setLabel("bottom", _xl, units=_xu)
        if payload_y_axis is not None:
            self.plot.setLabel("left", ylabel, units=y_unit)
        else:
            self.plot.setLabel("left", ylabel)

        return s_xdata, s_ydata

    # ── Waterfall rendering ───────────────────────────────────────

    def _waterfall_active(self):
        """Whether the bottom panel is currently rendered by ``wf_widget``."""
        try:
            n_curves = len(self.plot_data[1])
        except Exception:
            n_curves = 0
        return (
            (self.plotMethod == 'Waterfall' and n_curves > 3)
            or (self.plotMethod not in ('Sum', 'Average') and n_curves > 15)
        )

    @staticmethod
    def _uniform_waterfall_grid(xdata, data):
        """Return waterfall rows on an x grid compatible with ImageItem."""
        xdata = np.asarray(xdata, dtype=float)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or xdata.shape != (data.shape[1],):
            return xdata, data
        data, xdata = resample_image_axis_to_uniform(data, xdata, axis=1)
        return xdata, data

    def _frame_scan_info(self, idx):
        """Phase 3c: a frame's metadata (``scan_info``), store-first.

        Reads the publication's ``metadata_raw`` (the store holds every frame's
        1D publication in its unbounded light tier, so a whole-scan waterfall is
        covered), falling back to the in-memory ``frames`` browse cache then the
        legacy ``data_1d`` mirror.  Returns ``{}`` when nothing is found."""
        idx = int(idx)
        store = getattr(self, 'publication_store', None)
        if store is not None:
            pub = store.get(idx)
            if pub is not None and pub.metadata_raw:
                return pub.metadata_raw
        frames = getattr(self, 'frames', None)
        fr = frames.get(idx) if hasattr(frames, 'get') else None
        if fr is None:
            d1 = getattr(self, 'data_1d', None)
            fr = d1.get(idx) if hasattr(d1, 'get') else None
        return getattr(fr, 'scan_info', None) or {}

    def _wf_y_axis(self, n_rows: int):
        """Compute the waterfall y-axis array.

        G1: shared between :meth:`update_wf` and
        :meth:`update_wf_pmesh`.  Returns the ydata array shaped to
        ``n_rows`` (= data.shape[0] after the wf_start/wf_step slice)
        or ``None`` when a metadata key is missing — caller handles
        the no-axis case.

        ``self.wf_yaxis`` selects the source:

        * ``'Frame #'`` → ``arange`` rooted at ``wf_start + 1``.
        * ``'Time (s)'`` / ``'Time (minutes)'`` →
          ``scan_info['epoch']`` minus its own minimum (relative).
        * anything else → ``scan_info[wf_yaxis]`` directly.

        For everything but ``'Frame #'`` we lift the values from each frame's
        metadata via :meth:`_frame_scan_info` (store-first; Phase 3c) for the
        ACTUAL accumulated row ids, then slice with the same wf_start/wf_step the
        data uses.  The waterfall image rows come from the accumulator
        (``overlaid_idxs``), which after a preserved partial read or
        non-contiguous accumulation need NOT equal ``self.idxs`` -- so building the
        y-axis from ``self.idxs`` could drift the metadata values off the rendered
        rows.  Use ``overlaid_idxs`` (it is maintained row-for-row with
        ``plot_data``), falling back to ``_plot_row_ids`` / ``self.idxs``.
        """
        if self.wf_yaxis == 'Frame #':
            return np.asarray(np.arange(n_rows) + self.wf_start + 1,
                              dtype=float)
        row_ids = (getattr(self, 'overlaid_idxs', None)
                   or getattr(self, '_plot_row_ids', None)
                   or self.idxs)
        row_ids = [int(i) for i in row_ids]
        try:
            if self.wf_yaxis == 'Time (s)':
                s_ydata = np.asarray(
                    [self._frame_scan_info(idx)['epoch']
                     for idx in row_ids]
                )
                s_ydata -= s_ydata.min()
            elif self.wf_yaxis == 'Time (minutes)':
                s_ydata = np.asarray(
                    [self._frame_scan_info(idx)['epoch']
                     for idx in row_ids]
                ) / 60.
                s_ydata -= s_ydata.min()
            else:
                s_ydata = np.asarray(
                    [self._frame_scan_info(idx)[self.wf_yaxis]
                     for idx in row_ids]
                )
            return s_ydata[self.wf_start:self.wf_stop:self.wf_step]
        except KeyError as e:
            logger.debug('Counter %s not present in metadata: %s',
                         self.wf_yaxis, e)
            return None

    def update_wf(self):
        """Updates data in 1D plot Frame
        """
        # Throttle the expensive full-image waterfall REPAINT during a live scan.
        # The waterfall image grows O(N x M), and setImage re-copies + re-levels
        # (nanpercentile) + re-uploads the WHOLE stack every call -- the dominant
        # per-flush GUI-thread cost on a long scan (confirmed: gzip's slower frame
        # arrival makes the per-flush waterfall smaller -> smoother).  The
        # accumulator (plot_data) still grows every flush; we just repaint the
        # image at most ~2/sec while processing, so interaction stays responsive.
        # Idle (end-of-scan / user actions) always repaints so the final state is
        # complete and clicks feel instant.
        if getattr(self, "_processing_active", False):
            import time as _t
            now = _t.perf_counter()
            if now - getattr(self, "_wf_last_draw_t", 0.0) < 0.5:
                return
            self._wf_last_draw_t = now

        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start:self.wf_stop:self.wf_step, :]
        s_xdata, data = self._uniform_waterfall_grid(s_xdata, data)

        s_ydata = self._wf_y_axis(data.shape[0])
        if s_ydata is None:
            return

        from ...gui_utils import get_rect
        rect = get_rect(s_xdata, s_ydata)

        self.wf_widget.setImage(data.T, scale=self.scale, cmap=self.cmap)
        self.wf_widget.setRect(rect)

        _xl, _xu = self._current_plot_axis_label()
        self.wf_widget.image_plot.setLabel("bottom", _xl, units=_xu)
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

    def update_wf_pmesh(self):
        """Updates data in 1D plot Frame (pcolormesh waterfall rendering)
        """
        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start:self.wf_stop:self.wf_step, :]
        s_xdata, data = self._uniform_waterfall_grid(s_xdata, data)

        x_max, x_min = np.max(s_xdata), np.min(s_xdata)
        x_step = (x_max - x_min)/len(s_xdata)
        s_xdata = np.append(s_xdata, [x_max + x_step])
        s_xdata -= x_step/2
        s_xdata = np.tile(s_xdata, (data.shape[0]+1, 1)).T

        s_ydata = self._wf_y_axis(data.shape[0])
        if s_ydata is None:
            return

        y_max, y_min = np.max(s_ydata), np.min(s_ydata)
        y_step = (y_max - y_min)/len(s_ydata)
        s_ydata = np.append(s_ydata, [y_max + y_step])
        s_ydata -= y_step/2.
        s_ydata = np.tile(s_ydata, (data.shape[1]+1, 1))

        levels = np.nanpercentile(data, (1, 98))
        self.wf_widget.imageItem.setLevels(levels)
        self.wf_widget.imageItem.setData(s_xdata, s_ydata, data.T)
        self.wf_widget.imageItem.informViewBoundsChanged()

        from ...gui_utils import get_rect
        rect = get_rect(s_xdata[:, 0], s_ydata[0])
        self.wf_widget.setRect(rect)

        if getattr(self, 'viewer_mode', None) == 'xye':
            _xl, _xu = self._current_plot_axis_label()
        else:
            plotUnit = self.ui.plotUnit.currentIndex()
            _xl, _xu = x_labels_1D[plotUnit], x_units_1D[plotUnit]
        self.wf_widget.image_plot.setLabel("bottom", _xl, units=_xu)
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

        return data

    # ── Curve / layout helpers ────────────────────────────────────

    def setup_curves(self):
        """Initialize curves for line plots
        """
        self.curves.clear()
        self.legend.clear()

        frame_ids = self.frame_names[self.wf_start:self.wf_stop:self.wf_step]
        if (self.plotMethod in ['Sum', 'Average'] and
                len(self.frame_names) > 1):
            frame_ids = f'{self.plotMethod} [{self.frame_names[0]}'
            for frame_name in self.frame_names[1:]:
                frame_ids += f', {frame_name}'
            frame_ids = [frame_ids + ']']

        colors = self.get_colors()
        self.curves = [self.plot.plot(
            pen=color,
            symbolBrush=color,
            symbolPen=color,
            symbolSize=4,
            name=frame_id,
        ) for (color, frame_id) in zip(colors, frame_ids)]

        if not self.ui.showLegend.isChecked():
            self.legend.clear()

    def clear_1D(self):
        """Initialize curves for line plots
        """
        if hasattr(self, "clear_overlay"):
            self.clear_overlay()
        else:
            self.frame_names.clear()
            self.plot_data = [np.zeros(0), np.zeros(0)]
            self.plot_data_range = [[0, 0], [0, 0]]
            self.overlaid_idxs = []
        self.frame_ids.clear()
        self.setup_1d_layout()
        self.plot.clear()
        # Re-add legend (plot.clear() removes it)
        self.legend = self.plot.addLegend()
        # In a viewer mode the plotted curves ARE the file-list selection, so
        # Clear must reset to the pristine viewer state: reset the title (the
        # displayframe owns labelCurrent) and ask the host to drop the list
        # selection via sigCleared -- otherwise the cleared plot disagrees with a
        # stale selection + title that still names the old frames.  In integration
        # mode Clear only drops the accumulated overlay and KEEPS the selection so
        # it can be rebuilt, so this is scoped to viewer modes.
        if getattr(self, "viewer_mode", None) is not None:
            if hasattr(self, "_viewer_default_title"):
                self.ui.labelCurrent.setText(self._viewer_default_title())
            sig = getattr(self, "sigCleared", None)
            if sig is not None:
                sig.emit()

    def update_legend(self):
        if not self.ui.showLegend.isChecked():
            self.legend.hide()
        else:
            self.legend.show()

    def setup_1d_layout(self):
        """Setup the layout for 1D plot
        """
        self.wf_widget.setParent(None)
        self.plot_layout.addWidget(self.plot_win)
        if getattr(self, '_share_link_on', False):
            self._schedule_align()

        # Options is always reachable now — it holds Legend + Overlay Offset
        # (the waterfall y-axis/start/step inside grey out when not in a
        # waterfall).  Was disabled in single-curve mode, which would have
        # hidden the Legend toggle.
        self.ui.wf_options.setEnabled(True)
        if len(self.plot_data[1]) > 1:
            self.wf_yaxis_widget.setEnabled(False)

    def setup_wf_widget(self):
        self.plot_layout.addWidget(self.wf_widget)

        # Waterfall Plot setup
        if self.plotMethod == 'Waterfall':
            self.plot_win.setParent(None)
            self.plot_layout.addWidget(self.wf_widget)
            if getattr(self, '_share_link_on', False):
                self._schedule_align()
        else:
            self.wf_widget.setParent(None)
            self.plot_layout.addWidget(self.plot_win)
            if getattr(self, '_share_link_on', False):
                self._schedule_align()

    def setup_wf_layout(self):
        """Setup the layout for WF plot
        """
        self.plot_win.setParent(None)
        self.plot_layout.addWidget(self.wf_widget)
        if getattr(self, '_share_link_on', False):
            self._schedule_align()

        self.ui.wf_options.setEnabled(True)
        self.wf_yaxis_widget.setEnabled(True)

    # ── Waterfall options popup ───────────────────────────────────

    def popup_wf_options(self):
        """
        Popup Qt Window to select options for Waterfall Plot
        Options include Y-axis unit and number of points to skip
        """
        if self.wf_dialog.layout() is None:
            self.setup_wf_options_widget()
        elif self.wf_yaxis_widget.count() <= 3 and self.idxs_1d:
            # Dialog was first built with no trace loaded -- top up the
            # waterfall y-axis choices with the now-available metadata.
            info = self._frame_scan_info(self.idxs_1d[0])
            if info:
                self.wf_yaxis_widget.addItems(list(info))

        self.wf_dialog.show()

    def setup_wf_options_widget(self):
        """Build the 1D-plot Options dialog, grouped into three sections:

        * **Waterfall** — y-axis source (Frame #/Time/metadata), Start, Stop,
          Step.  Govern which frames map onto the waterfall image and its
          y-axis.
        * **Overlay** — Offset (vertical stacking between curves).
        * **Other** — legend toggle, intensity scale (Linear/Sqrt/Log),
          and colormap.
        """
        layout = QtWidgets.QVBoxLayout()
        self.wf_dialog.setLayout(layout)

        def _section(title):
            box = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout()
            box.setLayout(grid)
            layout.addWidget(box)
            return grid

        # ── Waterfall ─────────────────────────────────────────────
        wf = _section('Waterfall')
        wf.addWidget(QtWidgets.QLabel('Y-Axis'), 0, 0)
        wf.addWidget(QtWidgets.QLabel('Start'), 0, 1)
        wf.addWidget(QtWidgets.QLabel('Stop'), 0, 2)
        wf.addWidget(QtWidgets.QLabel('Step'), 0, 3)
        wf.addWidget(self.wf_yaxis_widget, 1, 0)
        wf.addWidget(self.wf_start_widget, 1, 1)
        wf.addWidget(self.wf_stop_widget, 1, 2)
        wf.addWidget(self.wf_step_widget, 1, 3)

        # ── Overlay ───────────────────────────────────────────────
        ov = _section('Overlay')
        ov.addWidget(QtWidgets.QLabel('Offset'), 0, 0)
        ov.addWidget(self.ui.yOffset, 0, 1)

        # ── Other ─────────────────────────────────────────────────
        # Legend toggle + the scale combo (moved out of the top bar;
        # re-parented here by addWidget).  The colormap lives in the top
        # bar next to the Log toggle.
        lg = _section('Other')
        lg.addWidget(self.ui.showLegend, 0, 0)
        lg.addWidget(self.ui.scale, 0, 1)

        # ── Dialog buttons ────────────────────────────────────────
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.wf_accept_button)
        btns.addWidget(self.wf_cancel_button)
        layout.addLayout(btns)

        # Options is reachable from launch now (it hosts scale/cmap), so a
        # trace may not exist yet -- fall back to the built-in counters; the
        # metadata counters are topped up on later opens (popup_wf_options).
        counters = ['Frame #', 'Time (s)', 'Time (minutes)']
        if self.idxs_1d:
            counters += list(self._frame_scan_info(self.idxs_1d[0]))
        self.wf_yaxis_widget.addItems(counters)

        self.wf_start_widget.setDecimals(0)
        self.wf_start_widget.setRange(1, 100000)
        self.wf_start_widget.setValue(1)

        # Stop: 0 is the sentinel for "through the last frame".
        self.wf_stop_widget.setDecimals(0)
        self.wf_stop_widget.setRange(0, 100000)
        self.wf_stop_widget.setValue(0)
        self.wf_stop_widget.setSpecialValueText('End')  # shown when value==0

        self.wf_step_widget.setDecimals(0)
        self.wf_step_widget.setRange(1, 1000)

        self.wf_accept_button.clicked.connect(self.get_wf_option)
        self.wf_cancel_button.clicked.connect(self.close_wf_popup)

    def get_wf_option(self):
        self.wf_yaxis = self.wf_yaxis_widget.currentText()

        self.wf_start = int(self.wf_start_widget.value()) - 1
        # Stop: 0 (shown as "End") → None = slice through the last frame;
        # otherwise it's a 1-based inclusive end (→ exclusive Python stop).
        _stop = int(self.wf_stop_widget.value())
        self.wf_stop = None if _stop <= 0 else _stop
        self.wf_step = int(self.wf_step_widget.value())

        self.close_wf_popup()
        self.update_plot_view()

    def close_wf_popup(self):
        self.wf_dialog.close()

    # ── Mouse tracking ────────────────────────────────────────────

    def trackMouse(self):
        """Set up mouse tracking on the plot scene.

        N6: pre-N6 this both passed ``slot=self.mouseMoved`` to the
        SignalProxy constructor AND called
        ``proxy.signal.connect(self.mouseMoved)`` afterwards —
        ``mouseMoved`` ran twice per mouse event.  Also: the proxy
        was a local variable that risked garbage-collection (Qt
        signals don't keep proxies alive on their own).  Stash it
        on ``self`` and use the constructor connection only.
        """
        self._mouse_proxy = pg.SignalProxy(
            signal=self.plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved,
        )

    def mouseMoved(self, pos):
        """Slot wired to ``pg.SignalProxy(sigMouseMoved)``.

        SignalProxy wraps emissions in a tuple — the slot receives
        ``(QPointF,)`` rather than the bare event.  Unpack defensively
        so the same slot also accepts a direct ``QPointF`` (in case
        a future caller wires up sigMouseMoved without the proxy).
        """
        from PySide6.QtCore import Qt as pyQt
        # SignalProxy delivers args as a single-element tuple; the
        # bare-signal path delivers the QPointF directly.  Handle both.
        if isinstance(pos, tuple):
            if not pos:
                return
            pos = pos[0]

        if len(self.curves) == 0:
            return

        if self.plot.sceneBoundingRect().contains(pos):
            vb = self.plot.vb
            self.plot_win.setCursor(pyQt.CrossCursor)
            mousePoint = vb.mapSceneToView(pos)
            self.pos_label.setText(f'x={mousePoint.x():.2f}, y={mousePoint.y():.2e}')
        else:
            self.pos_label.setText('')
            self.plot_win.setCursor(pyQt.ArrowCursor)

    # ── Slice overlay ─────────────────────────────────────────────

    def _set_slice_range(self, _=None, initialize=False):
        """Configure the slice label, range, and step size based on
        which axis is being sliced along.

        Uses ``self._plot_axis_info`` when available to determine the
        slice axis from the 2D integration metadata.  Falls back to the
        legacy index-based logic for standard mode.
        """
        idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= idx < len(self._plot_axis_info)
                else None)

        # Determine the slice axis label
        if info and info['source'] in ('2d', '1d_2d') and info.get('slice_axis'):
            slice_label = info['slice_axis']
        elif info and info['source'] == '1d':
            # 1D axis selected — default slice along Chi for standard mode
            slice_label = f'{Chi} ({Deg})'
        else:
            # Legacy fallback for standard mode
            if idx == 2:
                # Chi selected — slice along Q or 2Th
                if self.ui.imageUnit.currentIndex() != 1:
                    slice_label = f'Q ({AA_inv})'
                else:
                    slice_label = f'2{Th} ({Deg})'
            else:
                slice_label = f'{Chi} ({Deg})'

        # Extract the short label for display (strip units in parens)
        short_label = re.sub(r'\s*\(.*\)', '', slice_label).strip()

        # Configure ranges based on what we're slicing
        is_q_like = any(s in slice_label for s in (AA_inv, 'Q'))
        is_angle = any(s in slice_label for s in (Deg, Chi, Th, 'angle', 'Exit'))

        if is_q_like and not is_angle:
            # Q-type axis (Å⁻¹)
            self.ui.slice.setText(f'{short_label} Range')
            self.ui.slice_center.setRange(0, 25)
            self.ui.slice_width.setRange(0, 30)
            self.ui.slice_center.setSingleStep(0.1)
            self.ui.slice_width.setSingleStep(0.1)
            if initialize:
                self.ui.slice_center.setValue(2)
                self.ui.slice_width.setValue(0.5)
        else:
            # Angle-type axis (degrees)
            self.ui.slice.setText(f'{short_label} Range')
            self.ui.slice_center.setRange(-180, 180)
            self.ui.slice_width.setRange(0, 270)
            self.ui.slice_center.setSingleStep(1)
            self.ui.slice_width.setSingleStep(1)
            if initialize:
                self.ui.slice_center.setValue(0)
                self.ui.slice_width.setValue(10)

    def _update_slice_range(self):
        """Handle imageUnit changes that affect slice range units.

        In standard mode with χ selected as plotUnit, switching between
        Q-χ and 2θ-χ imageUnit requires converting the slice range
        between Q and 2θ.  In GI mode or when the axis metadata
        directly specifies the slice axis, just refresh the range.
        """
        if not self.ui.slice.isChecked():
            self.clear_slice_overlay()
            return

        plotUnit = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[plotUnit]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= plotUnit < len(self._plot_axis_info)
                else None)

        # In GI mode or when metadata explicitly defines the slice axis,
        # no unit conversion is needed — just refresh
        if self.scan.gi or (info and info['source'] not in ('2d', '1d_2d')):
            self.update_plot()
            return

        # Standard mode, chi axis: handle Q ↔ 2θ conversion
        if not self.scan.gi and info and info.get('axis') == 'azimuthal':
            imageUnit = self.ui.imageUnit.currentIndex()
            cen = self.ui.slice_center.value()
            wid = self.ui.slice_width.value()
            _range = np.array([cen - wid, cen + wid])

            # Phase 3c: wavelength is a scan constant, so source the frame for
            # _get_wavelength from the in-memory frames cache (not data_1d); it
            # falls back to the scan-level / NeXus wavelength when the frame
            # isn't resident, so no per-frame data_1d entry is needed.
            if not self.idxs_1d:
                self.update_plot()
                return
            frames = getattr(self, 'frames', None)
            frame_for_wl = (frames.get(self.idxs_1d[0])
                            if hasattr(frames, 'get') else None)
            wavelength = self._get_wavelength(frame_for_wl)
            if wavelength is None or wavelength <= 0:
                self.update_plot()
                return

            if imageUnit == 0:
                if self.ui.slice.text() == f'2{Th} Range':
                    _range = ((4 * np.pi / (wavelength * 1e10))
                              * np.sin(np.radians(_range / 2)))
            else:
                if self.ui.slice.text() == 'Q Range':
                    _range = (2 * np.degrees(
                        np.arcsin(_range * (wavelength * 1e10) / (4 * np.pi))))

            cen = (_range[-1] + _range[0]) / 2.
            wid = (_range[-1] - _range[0]) / 2.
            self.ui.slice_center.setValue(cen)
            self.ui.slice_width.setValue(wid)

        self._set_slice_range()
        self.show_slice_overlay()

    def show_slice_overlay(self):
        """
        Shows the slice integration region on 2D Binned plot.

        The overlay orientation depends on which axis the user is
        displaying (radial vs azimuthal):
        - Displaying radial → slicing along azimuthal → horizontal band
        - Displaying azimuthal → slicing along radial → vertical band
        """
        self.clear_slice_overlay()

        if not self.ui.slice.isChecked():
            return

        idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= idx < len(self._plot_axis_info)
                else None)

        # Only show overlay for 2D-derived axes
        if info and info['source'] not in ('2d', '1d_2d'):
            return

        center = self.ui.slice_center.value()
        width = self.ui.slice_width.value()
        _range = [center - width, center + width]

        binned_data, rect = self.binned_data

        if rect is None:
            return

        # Determine orientation from axis metadata
        axis_type = info.get('axis', 'radial') if info else 'radial'

        if axis_type == 'radial':
            # Displaying radial axis → slice band is along azimuthal (y-axis)
            _range = [max(rect.top(), _range[0]),
                      min(rect.top() + rect.height(), _range[1])]
            width = (_range[1] - _range[0]) / 2.
            self.overlay = ROI(
                [rect.left(), _range[0]], [rect.width(), 2 * width],
                pen=(255, 255, 255),
                maxBounds=rect
            )
        else:
            # Displaying azimuthal axis → slice band is along radial (x-axis)
            _range = [max(rect.left(), _range[0]),
                      min(rect.left() + rect.width(), _range[1])]
            width = (_range[1] - _range[0]) / 2.
            self.overlay = ROI(
                [_range[0], rect.top()], [2 * width, rect.height()],
                pen=(255, 255, 255),
                maxBounds=rect
            )

        self.binned_widget.imageViewBox.addItem(self.overlay, ignoreBounds=True)

    def clear_slice_overlay(self):
        """Clear the overlay that shows integration slice"""
        if self.overlay is not None:
            self.binned_widget.imageViewBox.removeItem(self.overlay)
            self.overlay = None
