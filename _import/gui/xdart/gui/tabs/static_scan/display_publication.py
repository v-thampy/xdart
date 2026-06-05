# -*- coding: utf-8 -*-
"""Display adapter for :mod:`xdart.modules.frame_publication`.

This module is GUI-local on purpose: ``PublicationStore`` itself stays
Qt-free and display-agnostic, while this adapter translates publications into
the existing ``DisplayPayload`` shapes used by the static-scan renderer.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from xdart.modules.frame_publication import (
    publication_has_1d_errors,
    publication_has_2d_errors,
)

from .display_logic import (
    Axis,
    ImagePayload,
    Mode,
    PanelRole,
    PlotPayload,
    RawSource,
    Trace,
    sentinel_mask,
    x_axis_for_unit,
    convert_2d_radial,
    is_gi_2d_units,
)
from .display_constants import AA_inv, Deg, Th, x_labels_2D, x_units_2D


def _label_key(label: Any) -> Any:
    try:
        return int(label)
    except (TypeError, ValueError):
        return label


def _display_unit_symbol(unit) -> str:
    """Axis-label display symbol for a raw integration-unit string.

    The GI reciprocal-space units (``qip_A^-1`` / ``qoop_A^-1`` / ``qz_A^-1`` /
    ``qtot_A^-1`` …) and any Å⁻¹ unit display as ``Å⁻¹``; angle units
    (``chi_deg``, ``exit_angle_deg``) as ``°``.  Anything else passes through.
    Fixes GI cake axes reading ``Q_ip (qip_A^-1)`` instead of ``Q_ip (Å⁻¹)`` —
    the *label* (Q_ip/Q_oop) is right; only the unit string was the raw key.
    Only reached for units ``x_axis_for_unit`` doesn't already resolve."""
    u = str(unit or "").lower()
    if "a^-1" in u or "angstrom" in u:
        return AA_inv
    if "deg" in u:
        return Deg
    return str(unit or "")


def _axis_for_publication(axis) -> Axis:
    label, unit = x_axis_for_unit(getattr(axis, "unit", ""))
    if label == "x" and getattr(axis, "label", None):
        label = axis.label
        unit = _display_unit_symbol(getattr(axis, "unit", ""))
    return Axis(label=label, unit=unit)


def _image_axis_for_publication(axis, *, fallback_label: str) -> Axis:
    if axis is None:
        return Axis(fallback_label, "")
    label, unit = x_axis_for_unit(getattr(axis, "unit", ""))
    if label == "x" and getattr(axis, "label", None):
        label = axis.label
        unit = _display_unit_symbol(getattr(axis, "unit", "") or "")
    values = getattr(axis, "values", None)
    return Axis(label=label, unit=unit, values=None if values is None else np.asarray(values, dtype=float))


def _two_d_axes_match(ref_view, view, *, rtol=1e-5, atol=1e-8) -> bool:
    """True when two FrameViews' 2D cakes share the same axis *identity* —
    same ``two_d_kind``, same axis units, and same axis values (within
    tolerance) — so they may be averaged together.

    Multi-frame cake display averages same-shaped ``intensity_2d`` arrays;
    without this, two frames with the same (nchi, nq) shape but different
    q/chi (or qip/qoop) axes would silently blend.  That same-shape /
    different-axis case is exactly the live↔batch↔reload drift the
    publication contract is meant to catch, so a mismatch returns False
    (the caller skips, never averages)."""
    if getattr(ref_view, "two_d_kind", None) != getattr(view, "two_d_kind", None):
        return False
    pairs = (
        (getattr(ref_view, "axis_2d_x", None), getattr(view, "axis_2d_x", None)),
        (getattr(ref_view, "axis_2d_y", None), getattr(view, "axis_2d_y", None)),
    )
    for ref_axis, axis in pairs:
        if ref_axis is None or axis is None:
            if ref_axis is not axis:
                return False
            continue
        if getattr(ref_axis, "unit", None) != getattr(axis, "unit", None):
            return False
        rv = np.asarray(getattr(ref_axis, "values", None), dtype=float)
        av = np.asarray(getattr(axis, "values", None), dtype=float)
        if rv.shape != av.shape or not np.allclose(
            rv, av, rtol=rtol, atol=atol, equal_nan=True
        ):
            return False
    return True


def _trace_name(publication, widget=None) -> str:
    scan = getattr(widget, "scan", None)
    scan_name = getattr(scan, "name", "")
    if scan_name and scan_name != "null_main":
        return f"{scan_name}_{publication.label}"
    source = (
        publication.metadata_raw.get("source_file")
        or publication.view.source_path
        or publication.source_identity
        or publication.label
    )
    if isinstance(source, str) and source:
        return os.path.basename(source)
    return str(publication.label)


def _current_axis_info(widget) -> dict[str, Any]:
    try:
        idx = int(widget.ui.plotUnit.currentIndex())
    except Exception:
        return {"source": "1d", "slice_axis": None, "axis": None}
    info = getattr(widget, "_plot_axis_info", ())
    if 0 <= idx < len(info):
        return dict(info[idx])
    return {"source": "1d", "slice_axis": None, "axis": None}


def _slice_enabled(widget) -> bool:
    try:
        return bool(widget.ui.slice.isChecked())
    except Exception:
        return False


def _canonical_plot_unit(widget) -> str:
    try:
        text = str(widget.ui.plotUnit.currentText()).lower()
    except Exception:
        text = ""
    if "2" in text and ("\u03b8" in text or "theta" in text or "th" in text):
        return "2th"
    if "\u03c7" in text or "chi" in text:
        return "chi"
    return "q"


def _canonical_axis_unit(axis) -> str:
    unit = str(getattr(axis, "unit", "") or "").lower()
    if "2th" in unit:
        return "2th"
    if "chi" in unit or unit in {"deg", "degrees"}:
        return "chi"
    if "q" in unit or "angstrom" in unit:
        return "q"
    return unit


class PublicationDisplayAdapter:
    """Resolve display payload fragments from a publication snapshot."""

    def __init__(self, store, *, widget=None):
        self._widget = widget
        self._items = {} if store is None else dict(store.snapshot())

    def available_1d_keys(self) -> set:
        return {
            _label_key(label)
            for label, publication in self._items.items()
            if publication.view.has_1d and not publication_has_1d_errors(publication)
        }

    def available_2d_keys(self) -> set:
        return {
            _label_key(label)
            for label, publication in self._items.items()
            if publication.view.has_2d and not publication_has_2d_errors(publication)
        }

    def raw_availability(self) -> dict:
        return {
            _label_key(label): {
                "has_raw": self._has_full_raw(publication),
                "has_thumbnail": self._has_thumbnail(publication),
            }
            for label, publication in self._items.items()
        }

    def raw_image(self, state):
        # Resolves the raw detector panel for the *integration* views (the
        # Int 2D raw image).  The Image Viewer does NOT use this path: it is a
        # raw detector-file browser that applies NO processing mask, background
        # subtraction or monitor normalization, so ``ImageViewerController``
        # owns its raw-preview payload directly (``_image_viewer_raw_payload``).
        # Routing the Image Viewer through here re-applied normalization +
        # background and, when that yielded a non-finite array, blanked the
        # panel (the reproducible Int 1D (XYE) -> Image Viewer blank).
        panel = state.panel(PanelRole.RAW_2D)
        if panel is None or not panel.has_data:
            return None

        accum = None
        count = 0
        for label in state.render_ids:
            publication = self._items.get(_label_key(label))
            if publication is None:
                continue
            data, source = self._raw_array(publication, panel.source)
            if data is None:
                continue
            data = sentinel_mask(data)
            if data.ndim != 2:
                continue
            bg = getattr(publication.raw_ref, "bg_raw", 0)
            if source is RawSource.RAW:
                data = self._apply_detector_mask(data, publication)
                data = self._subtract_if_shape_matches(data, bg, "raw frame background")
            data = self._normalize(data, publication.metadata_raw)
            if accum is None:
                accum = data
            elif accum.shape == data.shape:
                accum = accum + data
            else:
                continue
            count += 1

        if accum is None or count == 0:
            return None
        if state.overall and count != len(state.render_ids):
            return None

        image = accum / count
        image = self._subtract_if_shape_matches(
            image,
            getattr(self._widget, "bkg_map_raw", 0),
            "raw-image background",
        )
        if image.size == 0 or not np.isfinite(image).any():
            return None

        # Legacy raw rendering flipped the detector rows after transposing
        # for pyqtgraph.  ImagePayload itself is row/column oriented, and
        # display_frame_widget transposes every ImagePayload, so pre-flip
        # here to preserve the visible detector orientation exactly.
        image = np.asarray(image, dtype=float)[::-1, :]
        return ImagePayload(
            image=image,
            axis_x=Axis("x", "Pixels", values=np.arange(image.shape[1])),
            axis_y=Axis("y", "Pixels", values=np.arange(image.shape[0])),
        )

    def cake_image(self, state):
        panel = state.panel(PanelRole.CAKE_2D)
        if panel is None or not panel.has_data:
            return None

        accum = None
        count = 0
        axis_x = axis_y = None
        ref_view = None
        ref_publication = None
        for label in state.render_ids:
            publication = self._items.get(_label_key(label))
            if (
                publication is None
                or not publication.view.has_2d
                or publication_has_2d_errors(publication)
            ):
                continue
            view = publication.view
            data = np.asarray(view.intensity_2d, dtype=float)
            if data.ndim != 2:
                continue
            data = self._normalize(data, publication.metadata_raw)
            if accum is None:
                accum = data
                ref_view = view
                ref_publication = publication
                axis_x = _image_axis_for_publication(view.axis_2d_x, fallback_label="x")
                axis_y = _image_axis_for_publication(view.axis_2d_y, fallback_label="y")
            elif accum.shape == data.shape and _two_d_axes_match(ref_view, view):
                accum = accum + data
            else:
                # Same shape but a different 2D axis identity (two_d_kind /
                # axis units / axis values) is real live↔batch↔reload drift.
                # Averaging would blend e.g. qip/qoop with q/chi or two
                # different grids — the publication contract exists to catch
                # exactly this, so skip rather than silently blend.
                continue
            count += 1

        if accum is None or count == 0 or axis_x is None or axis_y is None:
            return None
        if state.overall and count != len(state.render_ids):
            return None

        image = accum / count
        background = getattr(self._widget, "bkg_2d", 0)
        background = self._cake_background_for_image(background, image)
        image = self._subtract_if_shape_matches(
            image,
            background,
            "2D-image background",
        )
        if image.size == 0 or not np.isfinite(image).any():
            return None
        axis_x = self._apply_image_unit_2d(axis_x, ref_view, ref_publication)
        return ImagePayload(image=image, axis_x=axis_x, axis_y=axis_y)

    def _apply_image_unit_2d(self, axis_x, ref_view, ref_publication):
        """Apply the 2D-unit (imageUnit) Q↔2θ toggle to the cake radial axis,
        exactly like the legacy ``get_xydata``.

        Default ``Q-χ`` (or a 2θ axis under ``2θ-χ``) is a **no-op** — the
        publication-derived axis is returned unchanged, so the normal cake
        render is byte-identical to before.  Only a genuine unit difference
        (e.g. ``2θ-χ`` selected over a Q-integrated cake) converts the radial
        *values* and relabels, so the toggle works on every render instead of
        only via the old direct ``update_binned`` redraw.  GI cakes are left
        verbatim (their imageUnit combo is disabled)."""
        widget = self._widget
        if (widget is None or axis_x is None or axis_x.values is None
                or ref_view is None):
            return axis_x
        data_unit = str(getattr(ref_view.axis_2d_x, "unit", "") or "")
        az_unit = str(getattr(getattr(ref_view, "axis_2d_y", None), "unit", "") or "")
        scan = getattr(widget, "scan", None)
        if getattr(scan, "gi", False) or is_gi_2d_units(data_unit, az_unit):
            return axis_x
        try:
            image_label = widget.ui.imageUnit.currentText()
        except Exception:
            return axis_x
        want_tth = Th in image_label
        want_q = AA_inv in image_label
        have_tth = "2th" in data_unit
        # Nothing to do when the selection already matches the data's unit.
        if not ((want_tth and not have_tth) or (want_q and have_tth)):
            return axis_x
        try:
            wavelength_m = widget._get_wavelength(
                getattr(ref_publication, "raw_ref", None))
        except Exception:
            wavelength_m = None
        new_values = convert_2d_radial(
            axis_x.values, data_unit=data_unit,
            want_tth=want_tth, want_q=want_q, wavelength_m=wavelength_m,
        )
        idx = 1 if want_tth else 0
        return Axis(label=x_labels_2D[idx], unit=x_units_2D[idx], values=new_values)

    def plot_payload(self, state):
        # Overlay/Waterfall still use the legacy accumulator until the
        # publication payload owns overlay history explicitly.
        if state.method in ("Overlay", "Waterfall"):
            return None
        if not self._can_use_native_1d_axis(state):
            return None

        traces = []
        axis = None
        ref_x = None
        for label in state.render_ids:
            publication = self._items.get(_label_key(label))
            if (
                publication is None
                or not publication.view.has_1d
                or publication_has_1d_errors(publication)
            ):
                continue
            view = publication.view
            x = np.asarray(view.axis_1d.values, dtype=float)
            y = np.asarray(view.intensity_1d, dtype=float)
            if x.shape != y.shape:
                continue
            y = self._normalize(y, publication.metadata_raw)
            if ref_x is None:
                ref_x = x
                axis = _axis_for_publication(view.axis_1d)
            elif x.shape != ref_x.shape or not np.allclose(x, ref_x, equal_nan=True):
                y = np.interp(ref_x, x, y)
                x = ref_x
            traces.append(Trace(label=_trace_name(publication, self._widget), x=x, y=y))

        if not traces or axis is None:
            return None
        return PlotPayload(axis_x=axis, traces=tuple(traces))

    def _can_use_native_1d_axis(self, state) -> bool:
        widget = self._widget
        if widget is None:
            return True
        if state.mode is not Mode.INT_1D and state.mode is not Mode.INT_2D:
            return False
        if getattr(getattr(widget, "scan", None), "gi", False):
            return False
        axis_info = _current_axis_info(widget)
        source = axis_info.get("source", "1d")
        if source == "2d":
            return False
        if source == "1d_2d" and _slice_enabled(widget):
            return False

        selected_unit = _canonical_plot_unit(widget)
        for label in state.render_ids:
            publication = self._items.get(_label_key(label))
            if publication is None or not publication.view.has_1d:
                continue
            if _canonical_axis_unit(publication.view.axis_1d) != selected_unit:
                return False
            return True
        return False

    def _normalize(self, data, metadata):
        widget = self._widget
        if widget is None or not hasattr(widget, "normalize"):
            return np.asarray(data, dtype=float)
        return widget.normalize(np.asarray(data, dtype=float), metadata)

    def _raw_array(self, publication, source):
        if source is RawSource.THUMBNAIL:
            data = publication.view.thumbnail
            if data is None:
                data = getattr(publication.raw_ref, "thumbnail", None)
            return data, RawSource.THUMBNAIL

        if source is RawSource.RAW:
            data = publication.view.raw
            if data is None:
                data = getattr(publication.raw_ref, "map_raw", None)
            if data is not None:
                return data, RawSource.RAW

        data = publication.view.thumbnail
        if data is None:
            data = getattr(publication.raw_ref, "thumbnail", None)
        if data is not None:
            return data, RawSource.THUMBNAIL
        return None, RawSource.NONE

    def _apply_detector_mask(self, data, publication):
        data = np.asarray(data, dtype=float).copy()
        masks = []
        mask = getattr(publication.raw_ref, "mask", None)
        if mask is not None:
            masks.append(mask)
        scan = getattr(self._widget, "scan", None)
        global_mask = getattr(scan, "global_mask", None)
        if global_mask is not None:
            masks.append(global_mask)
        if not masks:
            return data
        flat_masks = []
        for mask in masks:
            try:
                arr = np.asarray(mask)
            except (TypeError, ValueError):
                continue
            if arr.dtype == bool:
                # A boolean mask applies only when it matches the image shape;
                # a bool array of any other shape is NOT a flat-index mask — skip
                # it rather than coercing True/False into indices 1/0 (which
                # would silently NaN pixels 0 and 1).
                if arr.shape == data.shape:
                    data[arr] = np.nan
                continue
            try:
                flat_masks.append(np.asarray(arr, dtype=int).ravel())
            except (TypeError, ValueError):
                continue
        if flat_masks:
            flat = np.unique(np.concatenate(flat_masks))
            flat = flat[(flat >= 0) & (flat < data.size)]
            if flat.size:
                data[np.unravel_index(flat, data.shape)] = np.nan
        return data

    @staticmethod
    def _subtract_if_shape_matches(data, background, label):
        data = np.asarray(data, dtype=float)
        bg = np.asarray(background)
        if bg.shape == () or bg.shape == data.shape:
            return data - background
        return data

    @staticmethod
    def _cake_background_for_image(background, image):
        """Convert legacy pyFAI background orientation into FrameView space.

        ``displayFrameWidget.setBkg`` still captures the 2D background through
        ``get_frames_int_2d`` in pyFAI result orientation ``(radial,
        azimuthal)``. Publication cakes are displayed through FrameView in
        ``(axis_y, axis_x)`` orientation. Transpose array backgrounds before
        subtracting so a selected frame subtracts to zero instead of subtracting
        its vertically/axis-swapped copy.
        """
        bg = np.asarray(background)
        if bg.shape == ():
            return background
        if bg.T.shape == np.asarray(image).shape:
            return bg.T
        return background

    @staticmethod
    def _has_thumbnail(publication) -> bool:
        if publication.view.thumbnail is not None:
            return True
        return getattr(publication.raw_ref, "thumbnail", None) is not None

    @staticmethod
    def _has_full_raw(publication) -> bool:
        if publication.view.raw is not None:
            return True
        return getattr(publication.raw_ref, "map_raw", None) is not None

    @classmethod
    def _has_raw(cls, publication) -> bool:
        return cls._has_full_raw(publication) or cls._has_thumbnail(publication)


def publication_availability(store) -> tuple[set, set, dict]:
    """Return loaded-1D keys, loaded-2D/raw keys, and raw availability."""

    adapter = PublicationDisplayAdapter(store)
    return (
        adapter.available_1d_keys(),
        adapter.available_2d_keys(),
        adapter.raw_availability(),
    )
