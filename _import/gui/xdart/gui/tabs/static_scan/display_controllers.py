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
  loading, delegated to the headless ``ssrl_xrd_tools.io`` API
  (``classify_image_source`` / ``load_image_frame`` /
  ``load_processed_raw_or_thumbnail``) — xdart never opens HDF5 to guess.
"""

from __future__ import annotations

import logging

from .display_logic import (
    Mode, compute_display_state, build_payload, register_controller,
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
    "register_default_controllers",
]


def _data_snapshot(widget):
    """Snapshot loaded keys + per-frame raw/thumbnail availability under the
    data lock (a short, single-pass critical section)."""
    with widget.data_lock:
        loaded_1d = set(widget.data_1d.keys())
        loaded_2d = set(widget.data_2d.keys())
        raw_avail = {
            int(k): {
                'has_raw': v.get('map_raw') is not None,
                'has_thumbnail': v.get('thumbnail') is not None,
            }
            for k, v in widget.data_2d.items()
            if isinstance(v, dict)
        }
        store = getattr(widget, "publication_store", None)
        if store is not None:
            pub_1d, pub_2d, pub_raw = publication_availability(store)
            loaded_1d.update(pub_1d)
            loaded_2d.update(pub_2d)
            raw_avail.update(pub_raw)
    return loaded_1d, loaded_2d, raw_avail


class _BaseController:
    """Common state assembly; subclasses supply ``all_frame_index``/``gi``."""

    def _all_frame_index(self, widget):
        return [], False   # (all_frame_index, gi) — viewer default: no scan

    def compute_state(self, widget, mode):
        all_index, gi = self._all_frame_index(widget)
        loaded_1d, loaded_2d, raw_avail = _data_snapshot(widget)
        return compute_display_state(
            mode=mode,
            selected_ids=list(widget.frame_ids),
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
        adapter = (
            None if store is None
            else PublicationDisplayAdapter(store, widget=widget)
        )
        return build_payload(state, adapter)


class ScanDisplayController(_BaseController):
    """Int 1D / Int 2D — the integration view.  Reads the scan frame index
    so an Overall selection aggregates the whole scan."""

    def _all_frame_index(self, widget):
        import numpy as np
        with widget.scan.scan_lock:
            all_index = list(np.asarray(widget.scan.frames.index, dtype=int))
        gi = bool(getattr(widget.scan, 'gi', False))
        return all_index, gi


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
        from ssrl_xrd_tools.io import classify_image_source
        return classify_image_source(path)

    @staticmethod
    def load_processed_frame(path, frame_label):
        """Resolve a processed-``.nxs`` frame to raw (via its source pointer)
        or its dequantized thumbnail; returns a ``RawFrameResult`` recording
        which it returned (so a flat mask is never re-applied to a thumbnail)."""
        from ssrl_xrd_tools.io import load_processed_raw_or_thumbnail
        return load_processed_raw_or_thumbnail(path, frame_label)

    @staticmethod
    def load_raw_frame(path, frame_idx):
        """Load a genuine raw detector frame (master / tiff / eiger) by
        0-based index."""
        from ssrl_xrd_tools.io import load_image_frame
        return load_image_frame(path, frame_idx)


class XYEViewerController(_BaseController):
    """1D ``.xye`` overlay viewer.  Selection is *viewer* frame ids; the
    x-axis comes from the file prefix, not the integration-unit combo (§8)."""


# Singleton adapters (stateless) registered for each mode.
_SCAN = ScanDisplayController()
_IMAGE = ImageViewerController()
_XYE = XYEViewerController()


def register_default_controllers():
    """Register the core controllers into the open registry.  Idempotent —
    runs at import (below) and is safe to call again from widget ``__init__``."""
    register_controller(Mode.INT_1D, _SCAN)
    register_controller(Mode.INT_2D, _SCAN)
    register_controller(Mode.IMAGE_VIEWER, _IMAGE)
    register_controller(Mode.XYE_VIEWER, _XYE)


# Register on import so simply importing this module (or display_frame_widget)
# populates the registry — no construction order to get wrong.
register_default_controllers()
