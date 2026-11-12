# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports

# Other imports
import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)
from matplotlib import cm

# This module imports
from ..gui_utils import RectViewBox
from .imageWidgetUI import Ui_Form
from .imageWidgetUI_static import Ui_Form as Ui_Form_static

# Qt imports
import pyqtgraph as pg
from pyqtgraph import Qt, QtGui, QtCore
from pyqtgraph.Qt import QtWidgets
from pyqtgraph import ColorMap
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyqtgraph import getConfigOption


QFileDialog = QtWidgets.QFileDialog


# Detector ceilings that can dominate autoscale: the detector SATURATION
# ceiling (np.iinfo(dtype).max — 65535 for uint16, derived from the raw dtype,
# R3-D) and the uint32 dead/hot sentinel.  When the "Mask Saturated" toggle is
# OFF these stay finite in the image, so excluding them from the level
# POPULATION (never from the data) keeps a saturated block from dragging the
# upper colour level up to the ceiling.
from xrd_tools.core.invalid import (
    UINT32_CEILING as _UINT32_CEIL,
    integer_saturation_ceiling as _integer_saturation_ceiling,
)

#: legacy GUI fallback when the raw integer dtype was lost upstream (float
#: frame); core never hardcodes 65535.
_DEFAULT_SAT_CEIL = 65535.0
_LEVEL_SUBSAMPLE_STRIDE = 4
_LEVEL_SUBSAMPLE_MIN_PIXELS = 4096
_LEVEL_CACHE_TTL_S = 1.0


def _subsample_for_levels(arr, mask=None):
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.size > _LEVEL_SUBSAMPLE_MIN_PIXELS:
        slices = (slice(None, None, _LEVEL_SUBSAMPLE_STRIDE),
                  slice(None, None, _LEVEL_SUBSAMPLE_STRIDE))
        if arr.ndim > 2:
            slices = slices + (slice(None),) * (arr.ndim - 2)
        sampled = arr[slices]
        if mask is None:
            return sampled, True
        sampled_mask = np.asarray(mask)[slices]
        if np.any(sampled_mask):
            return (sampled, sampled_mask), True
    if mask is None:
        return arr, False
    return (arr, np.asarray(mask)), False


def _level_population(displayed, pop_mask):
    (sample, sample_mask), _ = _subsample_for_levels(displayed, pop_mask)
    return sample[sample_mask]


def _finite_minmax(arr):
    sample, sampled = _subsample_for_levels(arr)
    finite = np.isfinite(sample)
    if not np.any(finite) and sampled:
        sample = np.asarray(arr)
        finite = np.isfinite(sample)
    if not np.any(finite):
        return 0.0, 1.0
    values = sample[finite]
    return float(np.nanmin(values)), float(np.nanmax(values))


def _ceiling_safe_levels(
        displayed, raw, pct, *, expand_degenerate=False):
    """Robust ``nanpercentile`` levels for ``displayed``, excluding exact
    detector-ceiling pixels from the percentile population.

    The ceiling mask is taken from the UNTRANSFORMED ``raw`` (``displayed`` may
    already be log/sqrt-transformed in place by the caller), so the test is
    always against original counts.  The saturation ceiling is DERIVED from the
    raw integer dtype (R3-D — consistent with the integration path) rather than
    hardcoding 65535; a float ``raw`` whose dtype was lost upstream falls back
    to 65535.  Pixels are never modified — a saturated ceiling pixel still
    renders at the top of the colormap; only the autoscale population drops it.
    Falls back to the full finite population when every finite pixel sits at the
    ceiling (an all-/mostly-saturated frame), avoiding NaN levels and the
    'autoscale to data min/max' regression.
    """
    disp = np.asarray(displayed)
    finite = np.isfinite(disp)
    if not finite.any():
        return (0.0, 1.0)
    pop_mask = finite
    raw_arr = np.asarray(raw)
    sat_ceil = _integer_saturation_ceiling(raw_arr)
    if sat_ceil is None:
        sat_ceil = _DEFAULT_SAT_CEIL
    if raw_arr.shape == disp.shape:
        ceiling = np.isfinite(raw_arr) & (
            (raw_arr == sat_ceil) | (raw_arr >= _UINT32_CEIL))
        if ceiling.any() and (finite & ~ceiling).any():
            pop_mask = finite & ~ceiling
        del ceiling
    del finite
    population = _level_population(disp, pop_mask)
    # Boolean indexing made a private population. Reuse it for percentile's
    # partition instead of copying it: if the stride grid is entirely masked,
    # this fallback can contain almost the whole frame, not just p/16 pixels.
    # pop_mask already excludes nonfinite values; nanpercentile would allocate
    # another population-sized mask needlessly.
    lo, hi = np.percentile(population, pct, overwrite_input=True)
    # Sparse cakes can contain far fewer non-zero bins than the upper-tail
    # percentile (for example 49 positive bins in a 500x500 image).  In that
    # case both percentile endpoints are zero even though real intensity is
    # present, and pyqtgraph receives a degenerate colour range.  Only on that
    # rare path, inspect the full ceiling-safe population and fall back to its
    # finite extent.  Constant images deliberately keep their legacy equal
    # levels; there is no additional contrast to recover.
    if (expand_degenerate
            and not (np.isfinite(lo) and np.isfinite(hi) and hi > lo)):
        full_population = disp[pop_mask]
        full_population = full_population[np.isfinite(full_population)]
        if full_population.size:
            full_lo = float(np.nanmin(full_population))
            full_hi = float(np.nanmax(full_population))
            if full_hi > full_lo:
                lo, hi = full_lo, full_hi
    return (float(lo), float(hi))


class XDImageWidget(Qt.QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Setup range slider
        #self.rangeSlider = RangeSlider(self)
        #self.sliderLayout = Qt.QtWidgets.QVBoxLayout(self.ui.sliderFrame)
        #self.sliderLayout.addWidget(self.rangeSlider)

        # Image pane setup
        self.image_layout = Qt.QtWidgets.QHBoxLayout(self.ui.imageFrame)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(0)
        self.image_win = pg.GraphicsLayoutWidget()
        self.image_layout.addWidget(self.image_win)
        self.histogram = pg.HistogramLUTWidget(self.image_win)
        self.ui.toolLayout.addWidget(self.histogram, 1, 0, 1, 2)
        self.imageViewBox = RectViewBox()
        self.image_plot = self.image_win.addPlot(viewBox=self.imageViewBox)
        # Seaborn-style talk-context fonts; grid off for image plots.
        from xdart.gui.themes import apply_seaborn_plot_style
        apply_seaborn_plot_style(self.image_plot, grid=False)
        self.imageItem = pg.ImageItem(colormap=cm.get_cmap('viridis'))

        self.image_plot.addItem(self.imageItem)
        self.histogram.setImageItem(self.imageItem)
        self.histogram.vb.setMouseEnabled(x=False, y=False)
        self.raw_image = np.array(0)
        self.norm_image = np.array(0)
        self.displayed_image = np.array(0)
        starter = np.random.normal(50, 10, 256*256)
        starter.sort()
        self.setImage(starter.reshape((256, 256)))

        self.ui.logButton.toggled.connect(self.update_image)
        self.ui.cmapBox.currentIndexChanged.connect(self.set_cmap)
        self.ui.cmapBox.setCurrentIndex(1)
        self.set_cmap(1)

        self.show()

    def setImage(self, image, rect=None):
        self.raw_image = image[()]
        self.norm_image = normalize(self.raw_image)
        self.displayed_image = self.norm_image[()]
        self.update_image()
        if rect is not None:
            self.imageItem.setRect(rect)

    def setRect(self, rect):
        self.imageItem.setRect(rect)

    def invalidate_level_cache(self):
        """Forget autoscale state without clearing the displayed image."""
        self._level_cache = None
        self._level_scan_token = None

    def update_image(self, q=None):
        self.displayed_image = np.copy(self.raw_image)
        if self.ui.logButton.isChecked():
            self.displayed_image[self.displayed_image < 0] = 0
            minval = self.displayed_image[self.displayed_image > 0].min()
            self.displayed_image = np.log10(self.displayed_image/minval + 1)
            self.imageItem.setImage(self.displayed_image)
            self.histogram.item.axis.setScale(self.raw_image.max() /
                                              self.displayed_image.max())
        else:
            self.imageItem.setImage(self.displayed_image)
            self.histogram.item.axis.setScale(None)

    def set_cmap(self, index):
        self.histogram.item.gradient.loadPreset(
            self.ui.cmapBox.itemText(index)
        )


class pgImageWidget(Qt.QtWidgets.QWidget):
    def __init__(self, parent=None, lockAspect=False, raw=False):
        super().__init__(parent)
        self.ui = Ui_Form_static()
        self.ui.setupUi(self)

        # Some options for Raw Images
        self.raw = raw

        # Image pane setup
        self.image_layout = Qt.QtWidgets.QHBoxLayout(self.ui.imageFrame)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(0)

        self.image_win = pg.GraphicsLayoutWidget()
        self.image_layout.addWidget(self.image_win)
        self.imageViewBox = RectViewBox(lockAspect=lockAspect)
        self.image_plot = self.image_win.addPlot(viewBox=self.imageViewBox)
        from xdart.gui.themes import apply_seaborn_plot_style
        apply_seaborn_plot_style(self.image_plot, grid=False)
        self.imageItem = pgXDImageItem(raw=self.raw)
        self.image_plot.addItem(self.imageItem)

        # Make Label Item for showing position
        self.make_pos_label()

        self.histogram = pg.ColorBarItem(width=15)
        # Have ColorBarItem control colors of img and appear in 'plot':
        self.histogram.setImageItem(self.imageItem, insert_in=self.image_plot)

        self.raw_image = np.zeros(0)
        self.displayed_image = np.zeros(0)
        self._level_cache = None
        self._level_scan_token = None
        # Never shown here: the owner's layout shows it with its parent. Shown before it had one, it opened as its
        # own window for an instant, and on macOS that takes and returns the activation (live 2026-10-08).

    def make_pos_label(self, itemPos=(1, 0), parentPos=(1, 1), offset=(0, -20)):
        self.image_win.addItem(self.imageItem.pos_label)
        self.imageItem.pos_label.anchor(itemPos=itemPos, parentPos=parentPos, offset=offset)
        self.imageItem.pos_label.setFixedWidth(1)

    def setImage(self, image, rect=None,
                 scale='Linear', cmap='viridis',
                 **kwargs):
        self._level_scan_token = kwargs.pop("level_scan_token", None)
        self.raw_image = image[()]
        self.update_image(scale, cmap, **kwargs)
        if rect is not None:
            self.setRect(rect)      # through the diagnostic funnel

    def setRect(self, rect):
        self.imageItem.setRect(rect)

    def _display_array_for_scale(self, scale):
        if scale == 'Linear':
            return np.asarray(self.raw_image)
        # The transformed copy keeps a float32 display array float32; any other input is transformed in float64 as
        # before, so a float64 cake is never downcast (DISPLAY-FLOAT32-128).
        source = np.asarray(self.raw_image)
        return np.array(source, dtype=np.float32 if source.dtype == np.float32 else float, copy=True)

    def _cached_levels(
            self, scale, cmap, pct, data_range, *, expand_degenerate=False):
        raw = np.asarray(self.raw_image)
        displayed = np.asarray(self.displayed_image)
        key = (
            scale,
            cmap,
            tuple(pct),
            tuple(displayed.shape),
            str(displayed.dtype),
            str(raw.dtype),
            getattr(self, "_level_scan_token", None),
            tuple(data_range),
            bool(expand_degenerate),
        )
        now = time.perf_counter()
        cached = self._level_cache
        if (cached is not None and cached.get("key") == key
                and now - cached.get("time", 0.0) <= _LEVEL_CACHE_TTL_S):
            return cached["levels"], True
        if expand_degenerate:
            levels = _ceiling_safe_levels(
                displayed, raw, pct, expand_degenerate=True)
        else:
            levels = _ceiling_safe_levels(displayed, raw, pct)
        self._level_cache = {"key": key, "time": now, "levels": levels}
        return levels, False

    def percentile_levels(self, pair=None):
        """Map display percentiles through this image's existing finite, ceiling-safe population.

        Retain the mapped pair until pixels change; no image upload or source read.
        """
        pair = tuple(float(value) for value in (self._automatic_percentiles if pair is None else pair))
        cached = getattr(self, "_percentile_cache", None)
        if cached is not None and cached[0] == pair:
            return cached[1]
        levels, _hit = self._cached_levels(
            "display", "", pair, self._display_domain,
            expand_degenerate=self._expand_degenerate_levels)
        self._percentile_cache = (pair, levels)
        return levels

    def update_image(self, scale='Linear', cmap='viridis', **kwargs):
        perf = bool(os.environ.get("XDART_PERF"))
        t0 = time.perf_counter() if perf else 0.0
        self.displayed_image = self._display_array_for_scale(scale)
        # pyqtgraph turns downsampling on for an image wider than 32767 pixels and leaves it on; set it for each image,
        # so the next ordinary one is drawn pixel for pixel again (DISPLAY-FLOAT32-128).
        downsample = max(self.displayed_image.shape[:2], default=0) > 2**15 - 1
        self._percentile_cache = None
        t_copy = time.perf_counter() if perf else 0.0
        # All-NaN / empty images poison every nanpercentile below (RuntimeWarning
        # + NaN levels -> pyqtgraph autoscale weirdness).  Higher layers normally
        # block these, but the widget guards itself: show the raw image with no
        # explicit levels and bail.
        if (self.displayed_image.size == 0
                or not np.isfinite(self.displayed_image).any()):
            self.imageItem.setImage(self.displayed_image, autoDownsample=downsample)
            if perf:
                logger.info(
                    "[PERF] image_widget render: copy=%.0fms transform=0ms "
                    "levels=0ms setImage=%.0fms hist=0ms total=%.0fms "
                    "level_cache=skip",
                    (t_copy - t0) * 1000,
                    (time.perf_counter() - t_copy) * 1000,
                    (time.perf_counter() - t0) * 1000,
                )
            return

        cmap = 'viridis' if cmap == 'Default' else cmap
        cm = pg.colormap.getFromMatplotlib(cmap)  # prepare a linear color map
        self.histogram.axis.setLogMode(False)

        linear_pct = tuple(kwargs.pop("linear_percentiles", (2, 98)))
        expand_degenerate = bool(
            kwargs.pop("expand_degenerate_levels", False))
        self._expand_degenerate_levels = expand_degenerate
        if scale == 'Log':
            min_val = np.nanmin(self.displayed_image)
            # Display arithmetic at the copy's precision (DISPLAY-FLOAT32-128): in float32 a frame spanning more than
            # 3.4e38 overflows the shift, so its largest pixels become +inf (top of the scale), and below -2**24 the
            # shift's +1 is lost, so its smallest become -inf (bottom); float64 does the same past 1.8e308 and -2**53.
            with np.errstate(over="ignore", divide="ignore"):
                if min_val < 1:
                    self.displayed_image -= (min_val - 1)
                self.displayed_image = np.log10(self.displayed_image)

            pct = (0.1, 99.9)

            self.histogram.axis.setLogMode(True)
        elif scale == 'Sqrt':
            min_val = np.nanmin(self.displayed_image)
            if min_val < 0:
                img = np.sqrt(np.abs(self.displayed_image))
                img[self.displayed_image < 0] *= -1
                self.displayed_image = img
            else:
                self.displayed_image = np.sqrt(self.displayed_image)

            pct = (0.5, 99.9)

        else:
            # Linear autoscale percentiles are selected by the panel owner.
            # Detector previews use a tighter ceiling-safe policy than cakes.
            # Log/Sqrt are unchanged.
            # _ceiling_safe_levels additionally drops exact detector-ceiling
            # (65535 / uint32) pixels from the percentile population so an
            # unmasked saturated block (toggle OFF) can't blow out the scale.
            pct = linear_pct

            self.histogram.axis.setLogMode(False)

        self._automatic_percentiles = pct
        t_transform = time.perf_counter() if perf else 0.0
        # This min/max is also needed for the colorbar limits below. Include it
        # in the short-lived cache key so an async replacement or reused frame
        # label with a radically different intensity range cannot inherit stale
        # levels for the rest of the pause/scan. Identical re-renders still hit.
        low, high = _finite_minmax(self.displayed_image)
        self._display_domain = (low, high)
        levels, cache_hit = self._cached_levels(
            scale, cmap, pct, (low, high),
            expand_degenerate=expand_degenerate)
        if expand_degenerate:
            # The ordinary colorbar bound uses a stride sample.  A sparse cake
            # may enter the full-population degenerate fallback precisely
            # because that sample misses its few nonzero bins; ensure the
            # recovered levels are not immediately clamped back to the sampled
            # constant bound.
            low = min(low, levels[0])
            high = max(high, levels[1])
        t_levels = time.perf_counter() if perf else 0.0
        kwargs.setdefault("autoDownsample", downsample)
        self.imageItem.setImage(self.displayed_image, levels=levels, **kwargs)
        t_setimage = time.perf_counter() if perf else 0.0
        self.histogram.setColorMap(cm)
        # Use nan-aware min/max: detector/processed frames carry NaN-masked
        # pixels, and np.min/np.max return NaN there.  NaN colorbar limits make
        # setLevels() clamp the nanpercentile levels to [NaN, NaN], so the image
        # falls back to pyqtgraph autoscale (data min/max) instead of the
        # percentile range — the Image Viewer "scaled to min/max" symptom.
        self.histogram.lo_lim, self.histogram.hi_lim = low, high
        self.histogram.setLevels(values=levels)
        if perf:
            t_hist = time.perf_counter()
            logger.info(
                "[PERF] image_widget render: copy=%.0fms transform=%.0fms "
                "levels=%.0fms setImage=%.0fms hist=%.0fms total=%.0fms "
                "level_cache=%s",
                (t_copy - t0) * 1000,
                (t_transform - t_copy) * 1000,
                (t_levels - t_transform) * 1000,
                (t_setimage - t_levels) * 1000,
                (t_hist - t_setimage) * 1000,
                (t_hist - t0) * 1000,
                "hit" if cache_hit else "miss",
            )


class pgXDImageItem(pg.ImageItem):
    def __init__(self, parent=None, raw=False):
        super().__init__(parent)

        self.pos_label = pg.LabelItem(justify='right')
        self.raw = raw

    def render(self):
        # Upstream accepts either indices or a Boolean mask for transparentLocations.
        # Empty RSM slices can have millions of NaNs: two intp coordinates per
        # pixel cost much more than a mask. Keep sparse indices and delegate all
        # QImage/LUT conversion, downsampling and non-scalar/GPU paths upstream.
        if (self.image is not None and self.image.size and self._xp is np
                and self.image.ndim == 2 and self.image.dtype.kind == 'f'
                and not self.autoDownsample and self._imageNanLocations is None
                and (self._imageHasNans is None or self._imageHasNans)):
            self._imageHasNans = bool(np.isnan(self.image.min()))
            if self._imageHasNans:
                image = self.image.T if self.axisOrder == 'col-major' else self.image
                mask = np.isnan(image)
                index_bytes = np.count_nonzero(mask) * 2 * np.dtype(np.intp).itemsize
                self._imageNanLocations = mask if index_bytes > mask.nbytes else mask.nonzero()
        super().render()

    def hoverEvent(self, ev):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if ev.isExit():
            self.pos_label.setText('')
            return

        data = self.image
        pos = ev.pos()
        # i, j = pos.y(), pos.x()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, data.shape[0] - 1))
        j = int(np.clip(j, 0, data.shape[1] - 1))
        val = data[i, j]
        ppos = self.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        if self.raw:
            self.pos_label.setText(f'{x:0.0f}, {y:0.0f} [{val:.2e}]')
        else:
            self.pos_label.setText(f'{x:0.2f}, {y:0.2f} [{val:.2e}]')


def normalize(arr):
    """

    Parameters
    ----------
    arr : numpy.ndarray
    """
    minval = arr.min()
    maxval = arr.max()
    if maxval > minval:
        return (arr - minval) / (maxval - minval)
    else:
        return np.ones_like(arr)

