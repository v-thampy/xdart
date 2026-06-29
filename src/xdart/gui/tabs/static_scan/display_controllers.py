# -*- coding: utf-8 -*-
"""Mode controllers for the static-scan display (Stage 5).

Each controller owns one mode's *selection rules* and *loading lifecycle*
and turns the live widget state into a :class:`DisplayState` (and its
payload).  They are registered into the open controller registry in
``display_logic`` (``register_controller``) so the widget never branches on
mode to decide how to build state — it just asks ``controller_for(mode)``.

This is the seam the future Stitch/Fit/RSM modules plug into: each adds its
own controller + registers it, with no change to the dispatch core.

Design rules (plan §3 / §8):
- A controller is a thin, stateless adapter over the widget's data; it reads
  ``widget`` attributes and calls the pure ``compute_display_state``.
- **Viewer controllers never consult ``scan.frames`` or the integration-unit
  combo** — their selection is *viewer* frame ids.
- The :class:`ImageViewerController` owns image-source classification and
  loading, delegated to the headless ``xrd_tools.io`` API
  (``classify_image_source`` / ``load_image_frame`` /
  ``load_processed_raw_or_thumbnail``) — xdart never opens HDF5 to guess.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .display_logic import (
    Axis,
    DisplayPayload,
    ImagePayload,
    Mode,
    PlotPayload,
    Trace,
    compute_display_state,
    build_payload,
    register_controller,
    sentinel_mask,
    standalone_viewer_image,
    stitch_display_state,
    stitch_image_payload,
    stitch_plot_payload,
    xye_unit_from_filename,
    x_axis_for_unit,
)
from .display_publication import (
    PublicationDisplayAdapter,
    publication_availability,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ScanDisplayController",
    "ImageViewerController",
    "XYEViewerController",
    "NexusViewerController",
    "StitchDisplayController",
    "register_default_controllers",
]


def _label_key(label):
    try:
        return int(label)
    except (TypeError, ValueError):
        return label


def _candidate_labels(mode, selected_ids, all_frame_index):
    """Labels whose availability can affect this render.

    Most live updates draw Auto Last / Single and only need the selected frame.
    Whole-scan selections still ask for the whole scan so Sum/Average/Overall
    keep their full-coverage safety checks.
    """
    if mode in (Mode.INT_1D, Mode.INT_2D):
        try:
            if len(selected_ids) == len(all_frame_index) and len(all_frame_index) > 1:
                return tuple(all_frame_index)
        except TypeError:
            pass
    return tuple(selected_ids)


class _FrameIndexCount:
    """Length-only scan-index view for non-Overall renders.

    ``compute_display_state`` only needs the full frame labels when the current
    selection covers the whole scan.  Auto-last/Single live renders only need
    ``len(all_frame_index)`` to prove they are *not* Overall.  Carrying a
    length-only object avoids copying thousands of labels on every timer tick
    while still making accidental iteration obvious during tests.
    """

    def __init__(self, count):
        self._count = int(count)

    def __len__(self):
        return self._count

    def __iter__(self):
        raise RuntimeError("frame labels were requested from a count-only index")


def _data_snapshot(widget, *, labels=None, include_legacy=True):
    """Snapshot loaded keys + per-frame raw/thumbnail availability.

    X1 (Phase 3a): the PublicationStore is the primary source.  Viewer modes
    still opt into the legacy dict caches because they are row/file browsers
    rather than scan-display publications.  Normal Int 1D/2D modes do NOT
    OR-merge those mirrors into readiness anymore: doing so hid store eviction
    from the render decision and kept the old ``data_1d`` mirror as an implicit
    source of truth.
    """
    store = getattr(widget, "publication_store", None)
    loaded_1d, loaded_2d, raw_avail = set(), set(), {}
    if store is not None:
        pub_1d, pub_2d, pub_raw = publication_availability(store, labels=labels)
        loaded_1d |= pub_1d
        loaded_2d |= pub_2d
        raw_avail.update(pub_raw)
    if not include_legacy:
        return loaded_1d, loaded_2d, raw_avail

    label_filter = None if labels is None else {_label_key(label) for label in labels}
    with widget.data_lock:
        if label_filter is None:
            data_1d_keys = set(widget.data_1d.keys())
            data_2d_items = tuple(widget.data_2d.items())
        else:
            data_1d_keys = {key for key in label_filter if key in widget.data_1d}
            data_2d_items = tuple(
                (key, widget.data_2d[key])
                for key in label_filter
                if key in widget.data_2d
            )
        loaded_1d |= data_1d_keys
        loaded_2d |= {key for key, _value in data_2d_items}
        for k, v in data_2d_items:
            if not isinstance(v, dict):
                continue
            entry = raw_avail.setdefault(
                int(k), {'has_raw': False, 'has_thumbnail': False})
            entry['has_raw'] = bool(entry.get('has_raw')) or (
                v.get('map_raw') is not None)
            entry['has_thumbnail'] = bool(entry.get('has_thumbnail')) or (
                v.get('thumbnail') is not None)
    return loaded_1d, loaded_2d, raw_avail


def _image_viewer_raw_payload(widget, state):
    """Build the Image Viewer's raw-preview :class:`ImagePayload` (or ``None``).

    Mirrors the raw-browser semantics exactly (the behavior just fixed and
    verified in the GUI):

    * source: the selected frame's ``map_raw`` from ``data_2d``, falling back
      to its dequantized ``thumbnail`` when the full array isn't hydrated;
    * standalone files (``_viewer_is_xdart`` False): fill non-finite + the
      uint32 ceiling sentinel with the low finite value, **no** NaN mask;
    * processed-xdart files (``_viewer_is_xdart`` True): keep the baked NaN
      mask (``sentinel_mask``);
    * single-select — only ``render_ids[0]`` is shown (no overlay/accumulate);
    * **no** processing mask file or monitor normalization; a background is
      subtracted ONLY when the user set one in this mode (Set BG ->
      ``bkg_map_raw``, cleared on a mode change), resized to the displayed
      thumbnail -- a no-op for plain browsing.

    Returns ``None`` (→ the renderer clears the panel) when there is no
    selected frame, no ``map_raw``/``thumbnail``, or the sanitized array has no
    finite pixels.  The array is pre-flipped (``[::-1, :]``) because the
    renderer transposes every ``ImagePayload``; combined that reproduces the
    legacy ``data.T[:, ::-1]`` detector orientation, with ``Pixels`` axes.
    """
    if not state.render_ids:
        return None
    idx = int(state.render_ids[0])
    # X1 (Phase 3a): the publication is the primary render source; the
    # legacy data_2d mirror fills the gap while it still exists.
    raw = None
    store = getattr(widget, "publication_store", None)
    if store is not None:
        publication = store.get(idx)
        if publication is not None:
            raw = publication.view.raw
            if raw is None:
                raw = publication.view.thumbnail
    if raw is None:
        with widget.data_lock:
            frame_2d = widget.data_2d.get(idx)
        if frame_2d is not None:
            raw = frame_2d.get('map_raw')
            if raw is None:
                raw = frame_2d.get('thumbnail')
    if raw is None:
        return None
    if getattr(widget, '_viewer_is_xdart', False):
        # Image Viewer is raw-file inspection — never value-mask saturation
        # (baked xdart files already store NaN for true sentinels); the uint16
        # ceiling is left intact and the level-clamp keeps it from blowing out.
        data = sentinel_mask(raw, mask_saturation=False)
    else:
        data = standalone_viewer_image(raw)     # fill sentinels, no mask
    data = np.asarray(data, dtype=float)
    # Image Viewer Set BG: subtract the user-set background raw frame ONLY when its
    # shape matches the displayed frame (or it is a scalar) -- never resize a
    # possibly-incompatible background.  bkg_map_raw is 0 unless Set BG was used in
    # THIS mode (cleared on a mode change), so plain browsing is unaffected.
    _bkg = np.asarray(getattr(widget, "bkg_map_raw", 0))
    if _bkg.shape == () or _bkg.shape == data.shape:
        data = data - _bkg
    if data.ndim != 2 or data.size == 0 or not np.isfinite(data).any():
        return None
    image = data[::-1, :]
    return ImagePayload(
        image=image,
        axis_x=Axis("x", "Pixels", values=np.arange(image.shape[1])),
        axis_y=Axis("y", "Pixels", values=np.arange(image.shape[0])),
    )


class _BaseController:
    """Common state assembly; subclasses supply ``all_frame_index``/``gi``."""

    def _all_frame_index(self, widget):
        return [], False   # (all_frame_index, gi) — viewer default: no scan

    def compute_state(self, widget, mode):
        all_index, gi = self._all_frame_index(widget)
        selected_ids = list(widget.frame_ids)
        return self._compute_state_from_inputs(
            widget, mode, selected_ids=selected_ids, all_index=all_index, gi=gi)

    def _compute_state_from_inputs(self, widget, mode, *, selected_ids, all_index, gi):
        labels = _candidate_labels(mode, selected_ids, all_index)
        store = getattr(widget, "publication_store", None)
        loaded_1d, loaded_2d, raw_avail = _data_snapshot(
            widget,
            labels=labels,
            include_legacy=(
                store is None or mode not in (Mode.INT_1D, Mode.INT_2D)
            ),
        )
        return compute_display_state(
            mode=mode,
            selected_ids=selected_ids,
            all_frame_index=all_index,
            loaded_1d_keys=loaded_1d,
            loaded_2d_keys=loaded_2d,
            gi=gi,
            plot_unit='q_A^-1',          # affects only x_label; live axis via legacy path
            method=widget.ui.plotMethod.currentText(),
            unit_changed=False,
            prev_overlaid_ids=tuple(widget.overlaid_idxs),
            raw_availability=raw_avail,
            titles={},
            generation=widget.display_generation,
        )

    def build_payload(self, widget, state):
        store = getattr(widget, "publication_store", None)
        labels = tuple(dict.fromkeys((*state.selected_ids, *state.render_ids)))
        adapter = (
            None if store is None
            else PublicationDisplayAdapter(store, widget=widget, labels=labels)
        )
        return build_payload(state, adapter)


class ScanDisplayController(_BaseController):
    """Int 1D / Int 2D — the integration view.  Reads the scan frame index
    so an Overall selection aggregates the whole scan."""

    def _frame_index_count(self, widget):
        with widget.scan.scan_lock:
            count = len(widget.scan.frames.index)
        gi = bool(getattr(widget.scan, 'gi', False))
        return count, gi

    def _all_frame_index(self, widget):
        with widget.scan.scan_lock:
            all_index = [int(i) for i in widget.scan.frames.index]
        gi = bool(getattr(widget.scan, 'gi', False))
        return all_index, gi

    def compute_state(self, widget, mode):
        selected_ids = list(widget.frame_ids)
        count, gi = self._frame_index_count(widget)
        if mode in (Mode.INT_1D, Mode.INT_2D) and len(selected_ids) == count and count > 1:
            all_index, gi = self._all_frame_index(widget)
        else:
            all_index = _FrameIndexCount(count)
        return self._compute_state_from_inputs(
            widget, mode, selected_ids=selected_ids, all_index=all_index, gi=gi)


class ImageViewerController(_BaseController):
    """Raw 2D image viewer.  Selection is *viewer* frame ids — it never
    consults ``scan.frames``.  Owns image-source classification + loading,
    delegated to the headless ssrl API so xdart never opens HDF5 to guess
    what kind of file it is."""

    # ── image-source loading lifecycle (ssrl §5a) ──────────────────────

    @staticmethod
    def classify(path):
        """Classify an image file (raw master / processed-xdart /
        thumbnail-only / unknown) via the ssrl boundary."""
        from xrd_tools.io import classify_image_source
        return classify_image_source(path)

    @staticmethod
    def load_processed_frame(path, frame_label):
        """Resolve a processed-``.nxs`` frame to raw (via its source pointer)
        or its dequantized thumbnail; returns a ``RawFrameResult`` recording
        which it returned (so a flat mask is never re-applied to a thumbnail)."""
        from xrd_tools.io import load_processed_raw_or_thumbnail
        return load_processed_raw_or_thumbnail(path, frame_label)

    @staticmethod
    def load_raw_frame(path, frame_idx):
        """Load a genuine raw detector frame (master / tiff / eiger) by
        0-based index."""
        from xrd_tools.io import load_image_frame
        return load_image_frame(path, frame_idx)

    # ── raw-preview payload (Stage 4/5 step 2) ─────────────────────────
    def build_payload(self, widget, state):
        """Produce the Image Viewer's raw-preview payload directly.

        The Image Viewer is a raw detector-file browser, so its single panel
        is a ``RAW_2D`` :class:`ImagePayload` built straight from the selected
        frame's stored detector array — with NO processing mask, background
        subtraction or monitor normalization (those are integration concerns).
        This is the one render path for the mode; there is no fallback to a
        legacy ``_update_image_viewer``.
        """
        return DisplayPayload(
            generation=state.generation,
            raw_image=_image_viewer_raw_payload(widget, state),
            cake_image=None,
            plot=None,
        )


def _xye_plot_payload(widget, state):
    """Build the XYE viewer's :class:`PlotPayload` (or ``None``).

    One trace per selected frame (``render_ids``, in order — XYE is
    multi-select); the x-axis label/unit come from the *first* file's name
    prefix (``xye_unit_from_filename`` -> ``x_axis_for_unit``; unprefixed files
    default to Q, never an assumed 2θ), y-axis
    ``Intensity``; each trace labelled by its filename.  Returns ``None`` (-> the
    renderer clears the plot) on an empty selection or when no selected frame has
    1D data.

    selection == shown: the payload renders exactly the selected frames, so
    deselecting a file removes its curve immediately.  ``plotMethod``
    (Single/Overlay/Waterfall/Sum/Average) still controls *how* the selected
    curves are drawn; there is no lingering-after-deselect accumulation.
    """
    render_ids = []
    for i in state.render_ids:
        try:
            render_ids.append(int(i))
        except (TypeError, ValueError):
            continue
    if not render_ids:
        return None
    with widget.data_lock:
        frames = {i: widget.data_1d.get(i) for i in render_ids}

    store = getattr(widget, "publication_store", None)

    def _frame_data(idx):
        """(radial, intensity, source_file) — publication first (X1),
        legacy data_1d frame as the fallback while the dict exists."""
        if store is not None:
            pub = store.get(idx)
            if (pub is not None and pub.view.intensity_1d is not None
                    and pub.view.axis_1d is not None
                    and pub.view.axis_1d.values is not None):
                src = str(pub.metadata_raw.get('source_file', '') or '')
                return (np.asarray(pub.view.axis_1d.values, dtype=float),
                        np.asarray(pub.view.intensity_1d, dtype=float),
                        src)
        fr = frames.get(idx)
        int_1d = getattr(fr, 'int_1d', None) if fr is not None else None
        if int_1d is None:
            return None
        sinfo = getattr(fr, 'scan_info', None) or {}
        return (np.asarray(int_1d.radial, dtype=float),
                np.asarray(int_1d.intensity, dtype=float),
                str(sinfo.get('source_file', '') or ''))

    data = {i: _frame_data(i) for i in render_ids}

    # X-axis label from the first selected file's prefix (not a transform combo).
    first = data.get(render_ids[0])
    source_name = first[2] if first is not None else ''
    first_unit = xye_unit_from_filename(source_name)
    xlabel, xunits = x_axis_for_unit(first_unit)

    # Overlaying files of different known units mixes incompatible x-axes; we
    # label from the first file and warn so the user knows the axis isn't shared.
    if len(render_ids) > 1:
        units = set()
        for i in render_ids:
            d = data.get(i)
            u = xye_unit_from_filename(d[2] if d is not None else '')
            if u != 'unknown':
                units.add(u)
        if len(units) > 1:
            logger.warning(
                'XYE overlay mixes different x-axis units %s; labelling the '
                'axis from the first file (%s). Overlaid curves are on '
                'incompatible axes.', sorted(units), first_unit,
            )

    if first is None:
        return None

    # XYE Viewer Set BG: subtract the averaged background pattern, interpolated
    # onto each file's own grid (XYE files have per-file grids).  _bkg_xye is None
    # unless Set BG was used in this mode (cleared on a mode change).
    bkg_xye = getattr(widget, "_bkg_xye", None)
    bkg_x = bkg_y = None
    if bkg_xye is not None:
        bkg_x, bkg_y = bkg_xye
    traces = []
    for i in render_ids:
        d = data.get(i)
        if d is None:
            continue
        radial, intensity, src = d
        if bkg_x is not None and bkg_y is not None and np.size(bkg_x):
            intensity = intensity - np.interp(radial, bkg_x, bkg_y)
        traces.append(Trace(
            os.path.basename(src or f'xye_{i}'),
            radial,
            intensity,
        ))
    if not traces:
        return None

    return PlotPayload(
        axis_x=Axis(xlabel, xunits),
        traces=tuple(traces),
        axis_y=Axis('Intensity', ''),
    )


class XYEViewerController(_BaseController):
    """1D ``.xye`` overlay viewer.  Selection is *viewer* frame ids; the
    x-axis comes from the file prefix, not the integration-unit combo (§8)."""

    def build_payload(self, widget, state):
        """Render the XYE overlay through a :class:`PlotPayload` — the one
        render path for the mode (no legacy ``_update_xye_viewer`` fallback)."""
        return DisplayPayload(
            generation=state.generation,
            raw_image=None,
            cake_image=None,
            plot=_xye_plot_payload(widget, state),
        )


class NexusViewerController(_BaseController):
    """Read-only NeXus schema viewer.

    The actual HDF5 walking lives in ``xrd_tools.io.inspect_nexus``.
    This controller consumes the row preview published by ``H5Viewer``:
    1D previews draw as plots, 2D previews draw as bounded images, and
    metadata-only rows intentionally clear both panels.
    """

    def compute_state(self, widget, mode):
        loaded_1d, loaded_2d, raw_avail = set(), set(), {}
        with widget.data_lock:
            for key, frame in widget.data_1d.items():
                payload = getattr(frame, "nexus_preview_payload", None)
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("kind")
                if kind == "plot_1d":
                    loaded_1d.add(int(key))
                elif kind == "image_2d":
                    loaded_2d.add(int(key))
                    raw_avail[int(key)] = {
                        "has_raw": True,
                        "has_thumbnail": False,
                    }
        return compute_display_state(
            mode=mode,
            selected_ids=list(widget.frame_ids),
            all_frame_index=[],
            loaded_1d_keys=loaded_1d,
            loaded_2d_keys=loaded_2d,
            gi=False,
            plot_unit='q_A^-1',
            method=widget.ui.plotMethod.currentText(),
            unit_changed=False,
            prev_overlaid_ids=tuple(widget.overlaid_idxs),
            raw_availability=raw_avail,
            titles={},
            generation=widget.display_generation,
        )

    def build_payload(self, widget, state):
        if not state.render_ids:
            return DisplayPayload(
                generation=state.generation,
                raw_image=None,
                cake_image=None,
                plot=None,
            )
        idx = int(state.render_ids[0])
        with widget.data_lock:
            frame = widget.data_1d.get(idx)
        payload = getattr(frame, "nexus_preview_payload", None) if frame else None
        if not isinstance(payload, dict):
            return DisplayPayload(
                generation=state.generation,
                raw_image=None,
                cake_image=None,
                plot=None,
            )
        kind = payload.get("kind")
        if kind == "plot_1d":
            x = np.asarray(payload.get("x", ()), dtype=float)
            y = np.asarray(payload.get("y", ()), dtype=float)
            axis_x = Axis(
                str(payload.get("x_label") or "index"),
                str(payload.get("x_unit") or ""),
            )
            axis_y = Axis(
                str(payload.get("y_label") or "value"),
                str(payload.get("y_unit") or ""),
            )
            trace = Trace(str(payload.get("label") or idx), x=x, y=y)
            plot = PlotPayload(axis_x=axis_x, axis_y=axis_y, traces=(trace,))
            return DisplayPayload(
                generation=state.generation,
                raw_image=None,
                cake_image=None,
                plot=plot,
            )
        if kind == "image_2d":
            image = np.asarray(payload.get("image", ()), dtype=float)
            axis_x = Axis(
                str(payload.get("x_label") or "x"),
                str(payload.get("x_unit") or ""),
                values=np.asarray(payload.get("x", ()), dtype=float)
                if payload.get("x") is not None else None,
            )
            axis_y = Axis(
                str(payload.get("y_label") or "y"),
                str(payload.get("y_unit") or ""),
                values=np.asarray(payload.get("y", ()), dtype=float)
                if payload.get("y") is not None else None,
            )
            return DisplayPayload(
                generation=state.generation,
                raw_image=ImagePayload(image=image, axis_x=axis_x, axis_y=axis_y),
                cake_image=None,
                plot=None,
            )
        return DisplayPayload(
            generation=state.generation,
            raw_image=None,
            cake_image=None,
            plot=None,
        )


class StitchDisplayController(_BaseController):
    """Whole-scan stitch result (STITCH_1D / STITCH_2D).

    The merged result is a *synthetic* whole-scan record, not a per-frame
    publication, so this controller does NOT consult ``scan.frames``, the
    PublicationStore, or the integration-unit combo.  It reads the result
    straight off the scan (``scan.stitched_1d`` / ``scan.stitched_2d``, where
    ``ewald.stitch.run_stitch`` wrote it) and turns it into a single-panel
    :class:`DisplayState` + :class:`DisplayPayload`.  Making it a first-class
    display source is what lets the stitch survive subsequent ``update()`` calls
    (the previous one-shot ``render_stitch_result`` was overwritten by the next
    per-frame render)."""

    def compute_state(self, widget, mode):
        scan = getattr(widget, "scan", None)
        has_1d = getattr(scan, "stitched_1d", None) is not None
        has_2d = getattr(scan, "stitched_2d", None) is not None
        return stitch_display_state(
            mode, widget.display_generation, has_1d=has_1d, has_2d=has_2d)

    def build_payload(self, widget, state):
        scan = getattr(widget, "scan", None)
        plot = cake = None
        if state.mode is Mode.STITCH_2D:
            cake = stitch_image_payload(getattr(scan, "stitched_2d", None))
        else:
            plot = stitch_plot_payload(getattr(scan, "stitched_1d", None))
        return DisplayPayload(
            generation=state.generation,
            raw_image=None,
            cake_image=cake,
            plot=plot,
        )


# Singleton adapters (stateless) registered for each mode.
_SCAN = ScanDisplayController()
_IMAGE = ImageViewerController()
_XYE = XYEViewerController()
_NEXUS = NexusViewerController()
_STITCH = StitchDisplayController()


def register_default_controllers():
    """Register the core controllers into the open registry.  Idempotent —
    runs at import (below) and is safe to call again from widget ``__init__``."""
    register_controller(Mode.INT_1D, _SCAN)
    register_controller(Mode.INT_2D, _SCAN)
    register_controller(Mode.IMAGE_VIEWER, _IMAGE)
    register_controller(Mode.XYE_VIEWER, _XYE)
    register_controller(Mode.NEXUS_VIEWER, _NEXUS)
    register_controller(Mode.STITCH_1D, _STITCH)
    register_controller(Mode.STITCH_2D, _STITCH)


# Register on import so simply importing this module (or display_frame_widget)
# populates the registry — no construction order to get wrong.
register_default_controllers()
