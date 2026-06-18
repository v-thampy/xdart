# -*- coding: utf-8 -*-
"""Display adapter for :mod:`xdart.modules.frame_publication`.

This module is GUI-local on purpose: ``PublicationStore`` itself stays
Qt-free and display-agnostic, while this adapter translates publications into
the existing ``DisplayPayload`` shapes used by the static-scan renderer.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
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
    combine_flat_masks,
    nan_gaps_in_thumbnail,
    sentinel_mask,
    x_axis_for_unit,
    convert_2d_radial,
    is_gi_2d_units,
    nanmean_slice,
    resample_cake_to_unit,
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


def _label_keys(labels) -> tuple:
    if labels is None:
        return ()
    seen = set()
    keys = []
    for label in labels:
        key = _label_key(label)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def _display_ids_for_2d(state) -> tuple:
    """Frames the 2D cake / raw panel should render (Vivek's contract).

    Only Sum/Average AGGREGATE the whole selection; Single/Overlay/Waterfall show
    just the CURRENT (latest-selected) frame — lighter, and the cake/raw track the
    selected frame instead of an average.  ``render_ids`` is sorted ascending, so
    ``[-1]`` is the latest (the live Auto-Last frame, or the single browsed frame)."""
    ids = tuple(state.render_ids)
    if getattr(state, "method", None) in ("Sum", "Average"):
        return ids
    return ids[-1:]


class PublicationDisplayAdapter:
    """Resolve display payload fragments from a publication snapshot."""

    def __init__(self, store, *, widget=None, labels=None):
        self._widget = widget
        if store is None:
            self._items = {}
        elif labels is None:
            self._items = dict(store.snapshot())
        elif hasattr(store, "get_many"):
            self._items = store.get_many(_label_keys(labels))
        else:
            self._items = {
                label: publication
                for label in _label_keys(labels)
                if (publication := store.get(label)) is not None
            }

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
        mask_parts = []   # per-frame detector masks, for the thumbnail gap re-bake
        display_ids = _display_ids_for_2d(state)   # current frame, or all for Sum/Average
        for label in display_ids:
            publication = self._items.get(_label_key(label))
            if publication is None:
                continue
            data, source = self._raw_array(publication, panel.source)
            if data is None:
                continue
            # uint16-65535 masking is opt-in via the wrangler "Mask saturated"
            # toggle carried on the scan (default ON); non-finite + uint32 are
            # always masked.
            _scan = getattr(self._widget, "scan", None)
            data = sentinel_mask(
                data, mask_saturation=bool(getattr(_scan, "mask_sentinel", True)),
            )
            if data.ndim != 2:
                continue
            # Cache the full-resolution detector shape from any resident full
            # raw so a thumbnail-only render can map the flat gap mask into
            # thumbnail coordinates (parity with get_frames_map_raw and the
            # legacy update_image thumbnail path).
            if source is RawSource.RAW and self._widget is not None:
                try:
                    self._widget._raw_full_shape = tuple(data.shape)
                except Exception:
                    pass
            raw_ref = getattr(publication, "raw_ref", None)
            _frame_mask = getattr(raw_ref, "mask", None)
            if _frame_mask is not None:
                mask_parts.append(_frame_mask)
            bg = getattr(raw_ref, "bg_raw", getattr(raw_ref, "background", 0))
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
        if state.overall and count != len(display_ids):
            return None

        image = accum / count
        image = self._subtract_if_shape_matches(
            image,
            getattr(self._widget, "bkg_map_raw", 0),
            "raw-image background",
        )
        if image.size == 0 or not np.isfinite(image).any():
            return None

        # Detector module gaps are 0-valued (NOT sentinels), so sentinel_mask
        # never masks them.  _apply_detector_mask NaN'd them on the full-res path
        # above; the thumbnail source skips it, so bake the gap mask into a
        # downsampled image here -- mapping the flat detector indices into
        # thumbnail coordinates via the cached full-res shape -- so this payload
        # masks gaps identically to the full-res path and the legacy update_image
        # thumbnail path.  No-op for full-res (shape matches) or unknown shape.
        image = np.asarray(image, dtype=float)
        _scan = getattr(self._widget, "scan", None)
        # Authoritative full-res shape from the scan (persisted in the .nxs);
        # falls back to the live widget cache, then None.  Explicit is-None
        # checks (not truthiness) so a stray ndarray can't raise.
        full_shape = getattr(_scan, "detector_shape", None)
        if full_shape is None:
            full_shape = getattr(self._widget, "_raw_full_shape", None)
        gap_indices = combine_flat_masks(
            getattr(_scan, "global_mask", None),
            *mask_parts,
            size=(int(full_shape[0]) * int(full_shape[1]))
            if full_shape is not None else None,
        )
        if full_shape is not None and tuple(image.shape) != tuple(full_shape):
            nan_gaps_in_thumbnail(image, gap_indices, full_shape)

        # Legacy raw rendering flipped the detector rows after transposing
        # for pyqtgraph.  ImagePayload itself is row/column oriented, and
        # display_frame_widget transposes every ImagePayload, so pre-flip
        # here to preserve the visible detector orientation exactly.
        image = image[::-1, :]
        return ImagePayload(
            image=image,
            axis_x=Axis("x", "Pixels", values=np.arange(image.shape[1])),
            axis_y=Axis("y", "Pixels", values=np.arange(image.shape[0])),
            gap_mask_indices=gap_indices,
            raw_full_shape=tuple(full_shape) if full_shape is not None else None,
        )

    def cake_image(self, state):
        panel = state.panel(PanelRole.CAKE_2D)
        if panel is None or not panel.has_data:
            return None

        # The cake renders the CURRENT frame for Single/Overlay/Waterfall and the
        # aggregate only for Sum/Average (Vivek's contract, _display_ids_for_2d).
        display_ids = _display_ids_for_2d(state)

        # Eviction guard.  For Sum/Average the cake aggregates the WHOLE intended
        # set, so check store residency against selected_ids and serve the on-disk
        # aggregate (Step 7b) when a frame is evicted — else blank, never average a
        # wrong subset (review_2026-06-15 §2.C).  For Single/Overlay/Waterfall the
        # cake shows only the current frame, so guard just that one (None => blank;
        # the hydration worker rehydrates an evicted current frame and re-renders).
        guard_ids = (state.selected_ids
                     if state.method in ("Sum", "Average") else display_ids)
        for label in guard_ids:
            pub = self._items.get(_label_key(label))
            if pub is None or not pub.view.has_2d or publication_has_2d_errors(pub):
                return self._aggregate_cake_payload(state)

        accum = None
        count = 0
        axis_x = axis_y = None
        ref_view = None
        ref_publication = None
        for label in display_ids:
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
        if state.overall and count != len(display_ids):
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
        image, axis_x = self._apply_image_unit_2d(
            image, axis_x, ref_view, ref_publication)
        return ImagePayload(image=image, axis_x=axis_x, axis_y=axis_y)

    def _apply_image_unit_2d(self, image, axis_x, ref_view, ref_publication):
        """Apply the 2D-unit (imageUnit) Q↔2θ toggle to cake image + axis.

        Default ``Q-χ`` (or a 2θ axis under ``2θ-χ``) is a **no-op** — the
        publication-derived axis is returned unchanged, so the normal cake
        render is byte-identical to before.  Only a genuine unit difference
        (e.g. ``2θ-χ`` selected over a Q-integrated cake) resamples the image
        onto a grid uniform in the display unit and relabels the radial axis.
        A pyqtgraph ImageItem has one linear rect, so changing only the axis
        values would place interior peaks at the wrong 2θ.  GI cakes are left
        verbatim (their imageUnit combo is disabled)."""
        widget = self._widget
        if (widget is None or axis_x is None or axis_x.values is None
                or ref_view is None):
            return image, axis_x
        data_unit = str(getattr(ref_view.axis_2d_x, "unit", "") or "")
        az_unit = str(getattr(getattr(ref_view, "axis_2d_y", None), "unit", "") or "")
        scan = getattr(widget, "scan", None)
        if getattr(scan, "gi", False) or is_gi_2d_units(data_unit, az_unit):
            return image, axis_x
        try:
            image_label = widget.ui.imageUnit.currentText()
        except Exception:
            return image, axis_x
        want_tth = Th in image_label
        want_q = AA_inv in image_label
        have_tth = "2th" in data_unit
        # Nothing to do when the selection already matches the data's unit.
        if not ((want_tth and not have_tth) or (want_q and have_tth)):
            return image, axis_x
        try:
            wavelength_m = widget._get_wavelength(
                getattr(ref_publication, "raw_ref", None))
        except Exception:
            wavelength_m = None
        if not wavelength_m or wavelength_m <= 0:
            return image, axis_x
        image, new_values = resample_cake_to_unit(
            image,
            axis_x.values,
            data_unit=data_unit,
            want_tth=want_tth,
            want_q=want_q,
            wavelength_m=wavelength_m,
            axis=1,
        )
        idx = 1 if want_tth else 0
        return image, Axis(label=x_labels_2D[idx], unit=x_units_2D[idx],
                           values=new_values)

    def _aggregate_display_is_primary(self, dim: str) -> bool:
        widget = self._widget
        scan = getattr(widget, "scan", None)
        gate = getattr(widget, "_aggregate_display_is_primary", None)
        if scan is None or gate is None:
            return True
        try:
            return bool(gate(scan, dim))
        except Exception:
            return False

    def _aggregate_cake_payload(self, state):
        """Whole-scan (Overall) cake from the on-disk aggregate (Step 7b) when
        frames are evicted past the heavy store bound — the §2.C blank, filled.

        Returns ``None`` (blank) for a non-Overall selection, a non-Sum/Average
        method (Single/Overlay/Waterfall show the current frame, never the
        aggregate), a GI / non-primary mode, or until the off-thread aggregate is
        ready (it re-renders on completion).  Axes come from a resident frame's
        view when one exists (the frozen common grid makes them identical, and it
        carries the imageUnit Q↔2θ toggle); else from the aggregate's own q/χ."""
        if getattr(state, "method", None) not in ("Sum", "Average"):
            return None
        if not getattr(state, "overall", False):
            return None
        widget = self._widget
        if widget is None or not hasattr(widget, "_whole_scan_aggregate"):
            return None
        if not self._aggregate_display_is_primary("2d"):
            return ImagePayload(image=np.empty((0, 0), dtype=float))
        method = "sum" if state.method == "Sum" else "average"
        agg = widget._whole_scan_aggregate(dim="2d", method=method)
        if agg is None or agg.intensity is None:
            return None
        image = np.asarray(agg.intensity, dtype=float)   # (n_chi, n_q) — cake orient
        if image.ndim != 2:
            return None
        ref_view = ref_publication = None
        for label in state.render_ids:
            pub = self._items.get(_label_key(label))
            if (pub is not None and pub.view.has_2d
                    and not publication_has_2d_errors(pub)):
                ref_view, ref_publication = pub.view, pub
                break
        if ref_view is not None:
            axis_x = _image_axis_for_publication(ref_view.axis_2d_x, fallback_label="x")
            axis_y = _image_axis_for_publication(ref_view.axis_2d_y, fallback_label="y")
        else:
            axis_x = _image_axis_for_publication(
                SimpleNamespace(unit=agg.q_unit, label="Q", values=agg.q),
                fallback_label="Q")
            axis_y = _image_axis_for_publication(
                SimpleNamespace(unit=agg.chi_unit, label="χ", values=agg.chi),
                fallback_label="χ")
        background = getattr(widget, "bkg_2d", 0)
        background = self._cake_background_for_image(background, image)
        image = self._subtract_if_shape_matches(image, background, "2D-image background")
        if image.size == 0 or not np.isfinite(image).any():
            return None
        if ref_view is not None:
            image, axis_x = self._apply_image_unit_2d(
                image, axis_x, ref_view, ref_publication)
        else:
            image, axis_x = self._apply_image_unit_2d(
                image,
                axis_x,
                SimpleNamespace(axis_2d_x=axis_x, axis_2d_y=axis_y),
                SimpleNamespace(raw_ref=None),
            )
        return ImagePayload(image=image, axis_x=axis_x, axis_y=axis_y)

    def _aggregate_plot_payload(self, state):
        """Whole-scan 1D Sum/Average from the on-disk aggregate (Step 7b).

        This is the 1D counterpart to :meth:`_aggregate_cake_payload`: when an
        Overall Sum/Average selection is larger than the bounded heavy store,
        the display must aggregate the complete primary on-disk stack plus the
        unflushed tail, never the resident subset and never the legacy
        unbounded ``data_1d`` mirror.  The widget owns async dispatch/caching via
        ``_whole_scan_aggregate``; ``None`` means "not ready this render".
        """
        if getattr(state, "method", None) not in ("Sum", "Average"):
            return None
        if not getattr(state, "overall", False):
            return None
        widget = self._widget
        if widget is None or not hasattr(widget, "_whole_scan_aggregate"):
            return None
        if not self._aggregate_display_is_primary("1d"):
            return PlotPayload(axis_x=Axis(label="", unit=""), traces=())
        method = "sum" if state.method == "Sum" else "average"
        agg = widget._whole_scan_aggregate(dim="1d", method=method)
        if agg is None or agg.intensity is None:
            return None
        x = np.asarray(agg.q, dtype=float)
        y = np.asarray(agg.intensity, dtype=float)
        if x.shape != y.shape or x.size == 0 or not np.isfinite(y).any():
            return None
        axis = _axis_for_publication(
            SimpleNamespace(unit=agg.q_unit, label="Q", values=x)
        )
        scan = getattr(widget, "scan", None)
        scan_name = getattr(scan, "name", "") or "scan"
        return PlotPayload(
            axis_x=axis,
            traces=(Trace(label=f"{state.method} [{scan_name}]", x=x, y=y),),
        )

    def plot_payload(self, state):
        # Step 5 FLIP: the integration 1D plot now draws from the publication
        # record's active .view (see integration_plot_payload).  Single/Sum/
        # Average flow through it (Sum/Average collapse at render in
        # update_1d_view); Overlay/Waterfall + the 2D-slice/converted-axis
        # fallbacks return None there, so render_display falls back to the
        # legacy update_plot for those.
        if state.mode in (Mode.INT_1D, Mode.INT_2D):
            return self.integration_plot_payload(state)
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

    # ----------------------------------------------------------------- #
    # Full-parity integration 1D payload.  Step 5 WIRED this into
    # plot_payload (above) for INT_1D/INT_2D; Overlay/Waterfall + the
    # converted-axis-without-wavelength / 2D-slice cases return None here
    # and stay on the legacy update_plot via render_display's fallback.
    # ----------------------------------------------------------------- #
    def integration_plot_payload(self, state):
        """Build the INT_1D/INT_2D 1D :class:`PlotPayload` for the active-mode
        ``.view`` at legacy parity: native readout, GI verbatim axis, on-the-fly
        Q↔2θ, and 2D-slice-derived 1D projection.

        Returns ``None`` for Overlay/Waterfall (their cross-render history /
        waterfall-y state is not owned by the payload yet — a later step), and
        when no usable 1D trace can be built.  ``plot_payload`` calls this for
        integration modes and lets the legacy renderer handle ``None`` fallbacks."""
        if state.method in ("Overlay", "Waterfall"):
            return None
        widget = self._widget
        axis_info = _current_axis_info(widget)
        source = axis_info.get("source", "1d")
        needs_2d = (source == "2d") or (
            source == "1d_2d" and _slice_enabled(widget)
        )

        # Eviction parity (Step 7b): when any intended 1D frame is not resident
        # in the publication STORE, Overall Sum/Average should use the complete
        # on-disk aggregate.  Other non-resident selections return None so the
        # remaining legacy fallback/hydration path can handle explicit subsets
        # without silently dropping frames.  selected_ids holds the full intended
        # set for both overall (= all_frame_index) and explicit selections.
        for label in state.selected_ids:
            pub = self._items.get(_label_key(label))
            if pub is None:
                aggregate = (
                    self._aggregate_plot_payload(state) if not needs_2d else None
                )
                return aggregate
            if needs_2d:
                if not pub.view.has_2d or publication_has_2d_errors(pub):
                    return None
            elif not pub.view.has_1d or publication_has_1d_errors(pub):
                aggregate = self._aggregate_plot_payload(state)
                return aggregate

        traces = []
        axis = None
        ref_x = None
        for label in state.render_ids:
            publication = self._items.get(_label_key(label))
            if publication is None:
                continue
            view = publication.view
            if not needs_2d:
                if not view.has_1d or publication_has_1d_errors(publication):
                    continue
                x = np.asarray(view.axis_1d.values, dtype=float)
                y = np.asarray(view.intensity_1d, dtype=float)
                if x.shape != y.shape:
                    continue
                y = self._normalize(y, publication.metadata_raw)
                x, conv_axis = self._apply_plot_unit_1d(
                    x, str(getattr(view.axis_1d, "unit", "") or ""), publication)
                this_axis = (conv_axis if conv_axis is not None
                             else _axis_for_publication(view.axis_1d))
            else:
                projected = self._slice_1d_from_2d(view, publication, axis_info)
                if projected is None:
                    continue
                x, y, this_axis = projected
            if ref_x is None:
                ref_x = x
                axis = this_axis
            elif x.shape != ref_x.shape or not np.allclose(x, ref_x, equal_nan=True):
                y = np.interp(ref_x, x, y)
                x = ref_x
            traces.append(
                Trace(label=self._integration_trace_label(publication, axis_info),
                      x=x, y=y))

        if not traces or axis is None:
            return None
        return PlotPayload(axis_x=axis, traces=tuple(traces))

    def _apply_plot_unit_1d(self, x_values, data_unit, ref_publication):
        """1D analog of :meth:`_apply_image_unit_2d`: on-the-fly Q↔2θ for the
        ``plotUnit`` selector.  Returns ``(values, axis)`` — the converted
        values + the target :class:`Axis` when a conversion actually fires, else
        ``(x_values, None)`` (caller keeps the native axis).  GI reciprocal-space
        axes pass through verbatim; no wavelength ⇒ no conversion (and the native
        axis is kept, so the label never lies about un-converted values)."""
        widget = self._widget
        if widget is None or x_values is None:
            return x_values, None
        # Guard by UNIT, not the scan.gi flag: q_total (a |q| magnitude,
        # ``qtot_A^-1``/``q_A^-1``) IS Bragg-convertible to 2θ exactly like a
        # standard scan (legacy get_xdata converts it), while the signed/angle
        # GI axes (qip/qoop/exit) are not — is_gi_2d_units flags only the latter.
        if is_gi_2d_units(data_unit, ""):
            return x_values, None
        try:
            plot_label = widget.ui.plotUnit.currentText()
        except Exception:
            return x_values, None
        want_tth = Th in plot_label
        want_q = AA_inv in plot_label
        have_tth = "2th" in (data_unit or "")
        if not ((want_tth and not have_tth) or (want_q and have_tth)):
            return x_values, None
        try:
            wavelength_m = widget._get_wavelength(
                getattr(ref_publication, "raw_ref", None))
        except Exception:
            wavelength_m = None
        if not wavelength_m or wavelength_m <= 0:
            return x_values, None
        new_values = convert_2d_radial(
            x_values, data_unit=data_unit,
            want_tth=want_tth, want_q=want_q, wavelength_m=wavelength_m,
        )
        idx = 1 if want_tth else 0
        return new_values, Axis(label=x_labels_2D[idx], unit=x_units_2D[idx])

    def _slice_1d_from_2d(self, view, publication, axis_info):
        """Project a 1D curve from the active-mode 2D cake (the legacy
        ``get_int_1d`` 2D path), reducing over the slice axis.

        FrameView stores ``intensity_2d`` as ``(axis_2d_y=azimuthal,
        axis_2d_x=radial)`` whereas the legacy ``IntegrationResult2D.intensity``
        was ``(radial, azimuthal)`` — so the reduce-axis is FLIPPED here.  Only
        the radial display axis gets the Q↔2θ conversion (never χ)."""
        if not view.has_2d or publication_has_2d_errors(publication):
            return None
        intensity = np.asarray(view.intensity_2d, dtype=float)
        if intensity.ndim != 2:
            return None
        axis_type = axis_info.get("axis", "radial")
        if axis_type == "azimuthal":
            x_axis = view.axis_2d_y                       # χ / azimuthal
            slice_vals = getattr(view.axis_2d_x, "values", None)
            reduce_axis = 1                              # reduce over radial (FrameView axis 1)
            convert = False
        else:                                            # 'radial' (or fallback)
            x_axis = view.axis_2d_x                       # radial
            slice_vals = getattr(view.axis_2d_y, "values", None)
            reduce_axis = 0                              # reduce over azimuthal (FrameView axis 0)
            convert = True
        x = np.asarray(getattr(x_axis, "values", None), dtype=float)
        inds: Any = slice(None)
        if _slice_enabled(self._widget) and slice_vals is not None:
            slice_vals = np.asarray(slice_vals, dtype=float)
            try:
                center = float(self._widget.ui.slice_center.value())
                width = float(self._widget.ui.slice_width.value())
            except Exception:
                center = width = None
            if center is not None and width is not None and slice_vals.size:
                inds = (center - width <= slice_vals) & (slice_vals <= center + width)
        # nanmean_slice: None when the slice selects 0 bins, and no
        # "Mean of empty slice" warning on an all-NaN column (GI empty bins).
        if reduce_axis == 0:
            y = nanmean_slice(intensity[inds, :], 0)
        else:
            y = nanmean_slice(intensity[:, inds], 1)
        if y is None:
            return None
        y = self._normalize(y, publication.metadata_raw)
        this_axis = None
        if convert:
            x, this_axis = self._apply_plot_unit_1d(
                x, str(getattr(x_axis, "unit", "") or ""), publication)
        if this_axis is None:
            this_axis = _axis_for_publication(x_axis)
        if x.shape[0] != y.shape[0]:
            return None
        return x, y, this_axis

    def _integration_trace_label(self, publication, axis_info) -> str:
        """Trace name with the legacy slice-range suffix when slicing a 2D
        projection (``build_plot_names`` parity)."""
        name = _trace_name(publication, self._widget)
        if _slice_enabled(self._widget) and axis_info.get("source") in ("2d", "1d_2d"):
            try:
                c = float(self._widget.ui.slice_center.value())
                w = float(self._widget.ui.slice_width.value())
                # Match the legacy build_plot_names / _loaded_1d_overlay_labels
                # suffix EXACTLY: one decimal, U+00B1 (e.g. " [0.0±10.0]").
                name = f"{name} [{c:.1f}±{w:.1f}]"
            except Exception:
                pass
        return name

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
            if data is None:
                data = getattr(publication.raw_ref, "image", None)
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
        flat_data = data.reshape(-1)
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
                flat = np.asarray(arr, dtype=np.intp).ravel()
            except (TypeError, ValueError):
                continue
            flat = flat[(flat >= 0) & (flat < data.size)]
            if flat.size:
                # Duplicate indices are harmless; direct flat assignment avoids
                # a GUI-thread concatenate/unique/unravel pass on large masks.
                flat_data[flat] = np.nan
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
        return (
            getattr(publication.raw_ref, "map_raw", None) is not None
            or getattr(publication.raw_ref, "image", None) is not None
        )

    @classmethod
    def _has_raw(cls, publication) -> bool:
        return cls._has_full_raw(publication) or cls._has_thumbnail(publication)


def publication_availability(store, *, labels=None) -> tuple[set, set, dict]:
    """Return loaded-1D keys, loaded-2D/raw keys, and raw availability."""

    adapter = PublicationDisplayAdapter(store, labels=labels)
    return (
        adapter.available_1d_keys(),
        adapter.available_2d_keys(),
        adapter.raw_availability(),
    )
