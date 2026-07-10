# -*- coding: utf-8 -*-
"""Data fetching, processing, and export methods for displayFrameWidget.

This mixin extracts ~500 lines of data-access logic from the monolithic
displayFrameWidget class.  Methods here deal with reading data from
LiveScan / LiveFrame containers, normalization, unit conversion,
colour generation, and saving results to disk.

The mixin is designed to be inherited by displayFrameWidget alongside
QWidget, so all ``self`` references resolve to the composite widget.
"""

import logging
import os
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from .display_logic import RawSource, choose_raw_source, sentinel_mask
from xdart.modules.frame_publication import publication_from_frame_view
from xdart.modules.wavelength import normalize_wavelength_m, wavelength_angstrom_to_m
from xrd_tools.core import (
    DEFAULT_MODE_KEY,
    FrameRecord,
    FrameView,
    axis_from_unit,
    numeric_metadata,
    view_to_result_1d,
    view_to_result_2d,
)

logger = logging.getLogger(__name__)

# Sentinel for the MEM-1[14] store-first publication memo generation tracking
# (distinct from any real store generation, incl. None).
_STORE_FIRST_UNSET = object()

_BULK_1D_READ_CHUNK = 256


def _chunked(values, size):
    values = tuple(values)
    size = max(1, int(size))
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _norm_alias_key(name):
    text = str(name or '').strip()
    lower = text.lower()
    if lower in {'sec', 'second', 'seconds'}:
        return 'sec'
    if lower in {'monitor', 'mon'}:
        return 'Monitor'
    match = re.fullmatch(r'i(\d+)', lower)
    if match:
        return f'i{int(match.group(1))}'
    return None


def _norm_display_label(alias):
    return 'Monitor' if alias == 'Monitor' else alias


def _axes_close(a, b, *, rtol=1e-5, atol=1e-8):
    """Whether two 2D-cake axis arrays describe the SAME grid.

    The cross-frame cake reducer sums element-wise, so frames must share one
    q/chi grid.  Comparison is by value (same tolerance the stacked writer's
    uniform-axes check uses), not just shape — two different-integration-param
    cakes can share a shape while describing different grids.
    """
    if a is None or b is None:
        return a is None and b is None
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol)


def available_norm_channels(scan_data_keys):
    """Return normalization channels present in scan metadata.

    The display label is canonical/case-insensitive; the actual key preserves
    the column name that exists in the scan table or per-frame metadata.
    """
    seen = set()
    channels = []
    for key in scan_data_keys or ():
        alias = _norm_alias_key(key)
        if alias is None or alias in seen:
            continue
        seen.add(alias)
        channels.append((_norm_display_label(alias), key))
    return channels


class DisplayDataMixin:
    """Mixin providing data-fetching, processing, and export helpers.

    Expects the host widget to expose at least:

    - ``self.scan``, ``self.frame``, ``self.frames``
    - ``self.frame_ids``, ``self.viewer_rows_1d``, ``self.viewer_rows_2d``
    - ``self.data_lock`` (threading.RLock guarding the dicts)
    - ``self.idxs``, ``self.idxs_1d``, ``self.idxs_2d``, ``self.overall``
    - ``self.ui`` (the Ui_Form instance)
    - ``self.normChannel``, ``self.bkg_*``
    - ``self._plot_axis_info``

    Locking contract (M3)
    =====================

    ``viewer_rows_1d`` and ``viewer_rows_2d`` are shared between the wrangler
    thread, the integrator thread, the fileHandlerThread / the M1
    LoadFramesWorker, and the GUI thread.  All **mutating** access
    (assignment, ``del``, ``clear``, ``pop``) goes through
    ``self.data_lock``.  **Read** access either takes ``data_lock``
    explicitly (when iterating multiple keys) or uses
    :meth:`_snapshot_data` to grab a stable view to iterate
    lock-free.

    The CPython GIL makes single ``dict[k]`` lookups atomic, so
    direct cache-hit reads like ``self.viewer_rows_1d.get(idx)`` are still
    safe.  The unsafe pattern was *iterating* the dicts (``for k, v
    in self.viewer_rows_2d.items():`` or ``list(self.viewer_rows_2d.keys())``)
    without a lock — a concurrent writer can mutate the dict
    mid-iteration and raise RuntimeError ("dictionary changed size
    during iteration").  Use ``_snapshot_data`` for those paths.
    """

    @staticmethod
    def _sanitize_display_image(data, mask_saturation=True):
        """Return a float image with detector sentinels masked to NaN.

        Thin wrapper over the pure :func:`display_logic.sentinel_mask`
        (Stage 1 extraction); the masking logic is unit-tested headlessly.
        ``mask_saturation`` gates the (opt-in) uint16-65535 masking; non-finite
        + the uint32 ceiling are always masked.
        """
        return sentinel_mask(data, mask_saturation=mask_saturation)

    def _publication_from_store_for_display(self, idx, *, allow_blocking_read=None):
        """Return the scan-display publication for ``idx`` when available.

        Viewer modes keep using their row/file-browser caches.  Normal Int 1D/2D
        display paths read the bounded store/projection and queue hydration on a
        miss.
        """
        if getattr(self, "viewer_mode", None) is not None:
            return None
        store = getattr(self, "publication_store", None)
        if store is None:
            return None
        try:
            key = int(idx)
        except (TypeError, ValueError):
            key = idx
        async_enabled = bool(getattr(self, "_async_hydration_enabled", False))
        should_block = (
            (not async_enabled)
            if allow_blocking_read is None
            else bool(allow_blocking_read)
        )
        if should_block:
            return store.get_or_hydrate(key)
        publication = store.get(key)
        if publication is None:
            request = getattr(self, "_request_frame_hydration", None)
            if request is not None:
                try:
                    request(int(key))
                except Exception:
                    logger.debug(
                        "publication hydration request failed for %s",
                        key,
                        exc_info=True,
                    )
            return None
        # Do not synchronously open the NeXus file from the GUI thread.  Queue
        # full-payload hydration when the selected publication has been thinned.
        if getattr(publication, "raw_status", None) in ("evicted", "thumbnail"):
            request = getattr(self, "_request_frame_hydration", None)
            if request is not None:
                try:
                    request(int(key))
                except Exception:
                    logger.debug(
                        "publication hydration request failed for %s",
                        key,
                        exc_info=True,
                    )
        return publication

    def _display_hydration_should_block(self, allow_blocking_read=None) -> bool:
        """Whether a display fallback may synchronously hydrate from disk."""
        async_enabled = bool(getattr(self, "_async_hydration_enabled", False))
        return (
            (not async_enabled)
            if allow_blocking_read is None
            else bool(allow_blocking_read)
        )

    def _request_missing_publication(self, idx, *, purpose="full") -> None:
        request = getattr(self, "_request_frame_hydration", None)
        if request is None:
            return
        try:
            request(int(idx), purpose=purpose)
        except TypeError:
            try:
                request(int(idx))
            except Exception:
                logger.debug("publication hydration request failed for %s", idx,
                             exc_info=True)
        except Exception:
            logger.debug("publication hydration request failed for %s", idx,
                         exc_info=True)

    def _selected_publication_views(self, publication):
        """Return the current 1D/2D mode views from a publication record."""
        record = getattr(publication, "record", None)
        if record is None:
            view = publication.view
            return (view if view.has_1d else None, view if view.has_2d else None)

        scan = getattr(self, "scan", None)
        display_gi = getattr(self, "_display_gi_enabled", None)
        is_gi = (
            bool(display_gi(scan))
            if callable(display_gi) and scan is not None
            else bool(getattr(scan, "gi", False))
        )
        mode_1d = mode_2d = None
        if is_gi:
            args_getter = getattr(self, "_display_bai_args", None)
            a1 = (
                args_getter(scan, "1d")
                if callable(args_getter)
                else getattr(scan, "bai_1d_args", {}) or {}
            )
            a2 = (
                args_getter(scan, "2d")
                if callable(args_getter)
                else getattr(scan, "bai_2d_args", {}) or {}
            )
            mode_1d = a1.get("gi_mode_1d", "q_total")
            mode_2d = a2.get("gi_mode_2d", "qip_qoop")
        if is_gi:
            view_1d = record.view_1d(mode_1d)
            view_2d = record.view_2d(mode_2d)
        else:
            view_1d = record.view_1d()
            view_2d = record.view_2d()
        return view_1d, view_2d

    @staticmethod
    def _first_present(*values):
        for value in values:
            if value is not None:
                return value
        return None

    def _publication_legacy_parts(self, publication):
        """Adapt a :class:`FramePublication` into the legacy display shapes.

        The old renderer expects a LiveFrame-ish 1D object plus a 2D dict.  This
        keeps the renderer stable while changing only the data source.
        """
        view_1d, view_2d = self._selected_publication_views(publication)
        merged_view = getattr(publication, "view", None)
        raw_ref = getattr(publication, "raw_ref", None)
        metadata = dict(
            getattr(publication, "metadata_raw", None)
            or getattr(merged_view, "metadata_raw", None)
            or {}
        )

        int_1d = view_to_result_1d(view_1d) if view_1d is not None else None
        int_2d = view_to_result_2d(view_2d) if view_2d is not None else None
        thumbnail = self._first_present(
            getattr(view_2d, "thumbnail", None),
            getattr(view_1d, "thumbnail", None),
            getattr(merged_view, "thumbnail", None),
            getattr(raw_ref, "thumbnail", None),
        )
        raw = self._first_present(
            getattr(merged_view, "raw", None),
            getattr(view_2d, "raw", None),
            getattr(view_1d, "raw", None),
            getattr(raw_ref, "map_raw", None),
            getattr(raw_ref, "image", None),
        )
        frame_1d = None
        if (
            int_1d is not None
            or int_2d is not None
            or thumbnail is not None
            or raw_ref is not None
        ):
            frame_1d = SimpleNamespace(
                idx=publication.label,
                int_1d=int_1d,
                scan_info=metadata,
                thumbnail=thumbnail,
                map_raw=raw,
                integrator=getattr(raw_ref, "integrator", None),
                poni=getattr(raw_ref, "poni", None),
            )
        frame_2d = {
            "map_raw": raw,
            "bg_raw": getattr(raw_ref, "bg_raw", 0) if raw_ref is not None else 0,
            "mask": getattr(raw_ref, "mask", None) if raw_ref is not None else None,
            "int_2d": int_2d,
            "gi_2d": {},
            "thumbnail": thumbnail,
        }
        if int_2d is None and raw is None and thumbnail is None:
            frame_2d = {}
        return frame_1d, frame_2d

    def _display_publication_from_view(self, idx, view):
        """Wrap a selected ``FrameView`` in the GUI publication envelope.

        The session store owns integration arrays; GUI-only raw artifacts may
        still live in the PublicationStore.  Borrow raw / thumbnail / raw_ref
        from the publication fallback when the store view is intentionally
        integration-only, preserving current raw-panel behavior.
        """
        if view is None:
            return None
        try:
            key = int(idx)
        except (TypeError, ValueError):
            key = idx
        publication = None
        store = getattr(self, "publication_store", None)
        if store is not None:
            try:
                publication = store.get(key)
            except Exception:
                logger.debug("publication artifact lookup failed for %s", key,
                             exc_info=True)
        fallback_view = getattr(publication, "view", None)
        updates = {}
        if fallback_view is not None:
            if view.raw is None and getattr(fallback_view, "raw", None) is not None:
                updates["raw"] = fallback_view.raw
            if (
                view.thumbnail is None
                and getattr(fallback_view, "thumbnail", None) is not None
            ):
                updates["thumbnail"] = fallback_view.thumbnail
                updates["mask_baked"] = bool(
                    view.mask_baked or getattr(fallback_view, "mask_baked", False)
                )
        if updates:
            view = replace(view, **updates)
        return publication_from_frame_view(
            view,
            generation=int(getattr(self, "display_generation", 0) or 0),
            raw_ref=getattr(publication, "raw_ref", None),
            raw_status=getattr(publication, "raw_status", "unknown"),
            validate=True,
        )

    def _store_first_publication_for_display(
            self, idx, *, allow_blocking_read=False):
        lookup = getattr(self, "store_first_frame_view", None)
        if lookup is None:
            return None
        # MEM-1[14]: the live render resolves EVERY selected label per tick
        # (display_controllers._data_snapshot + _store_first_publication_items).
        # On an auto-last Overlay run the selection is the whole scan, so this
        # rebuilds O(N) publications (view projection + validate) per tick -- the
        # end-of-run ~3s main-thread freeze.  Memoize the build keyed by the
        # backing store record/publication IDENTITY (held by reference so an
        # id() can't be reused; changes only when a frame is re-published or
        # hydrated) plus the active projection modes, so only NEW/changed labels
        # rebuild -> O(k) per tick.  The generation stamp is re-applied cheaply on
        # a hit (replace() does not re-validate).  Blocking reads (user gestures)
        # bypass the cache and always resolve fresh.  The cache is dropped whenever
        # the store generation changes (scan boundary / eviction).
        if not allow_blocking_read and callable(
                getattr(self, "_store_first_cache_ident", None)):
            rec, pub, m1, m2 = self._store_first_cache_ident(idx)
            if rec is not None or pub is not None:
                cache = self._store_first_pub_cache()
                key = self._coerce_frame_label(idx)
                hit = cache.get(key)
                if (hit is not None and hit[0] is rec and hit[1] is pub
                        and hit[2] == m1 and hit[3] == m2):
                    built = hit[4]
                    gen = int(getattr(self, "display_generation", 0) or 0)
                    if getattr(built, "generation", None) != gen:
                        built = replace(built, generation=gen)
                    self._pub_memo_hits = getattr(self, "_pub_memo_hits", 0) + 1
                    return built
                view = lookup(idx, allow_blocking_read=False)
                if view is None:
                    return None
                built = self._display_publication_from_view(idx, view)
                if built is not None:
                    cache[key] = (rec, pub, m1, m2, built)
                self._pub_memo_misses = getattr(self, "_pub_memo_misses", 0) + 1
                return built
        view = lookup(idx, allow_blocking_read=bool(allow_blocking_read))
        if view is None:
            return None
        return self._display_publication_from_view(idx, view)

    @staticmethod
    def _coerce_frame_label(idx):
        """Store label for a display idx — same coercion staticWidget uses.
        The memo's original form assumed this existed on the displayframe (it
        was staticWidget-only); every ident lookup raised AttributeError into
        the silent except and the memo never engaged (MEM-1[14] follow-up)."""
        try:
            return int(idx)
        except (TypeError, ValueError):
            return idx

    def _store_first_pub_cache(self) -> dict:
        """Per-label memo for :meth:`_store_first_publication_for_display`
        (MEM-1[14]).  Reset whenever the publication store's generation changes
        (scan boundary / eviction) so stale cross-epoch entries can't survive."""
        store = getattr(self, "publication_store", None)
        gen = getattr(store, "generation", None) if store is not None else None
        if getattr(self, "_store_first_pub_cache_gen", _STORE_FIRST_UNSET) != gen:
            self._store_first_pub_cache_d = {}
            self._store_first_pub_cache_gen = gen
        return self._store_first_pub_cache_d

    def _store_first_cache_ident(self, idx):
        """Cheap per-label invalidation identities: the backing FrameRecord and
        FramePublication objects (compared by identity) + the active projection
        modes.  Returns ``(None, None, m1, m2)`` when the frame is absent.

        NB the record store is reached through ``self.frame_record_store`` —
        the displayframe's provider attribute (staticWidget assigns its bound
        ``_active_frame_record_store`` accessor there; same idiom as
        ``_hydration_stores``).  The memo's original form looked up
        ``_active_frame_record_store`` on ``self`` — a STATICWIDGET attribute
        that does not exist on the displayframe — so the identity was always
        ``(None, None)`` and the memo never engaged on the real widget
        (test-green/prod-dead, MEM-1[14] follow-up)."""
        try:
            key = self._coerce_frame_label(idx)
        except Exception:
            return None, None, None, None
        rec = None
        rec_store = getattr(self, "frame_record_store", None)
        if callable(rec_store):
            try:
                rec_store = rec_store()
            except Exception:
                rec_store = None
        if rec_store is not None:
            try:
                rec = rec_store.get(key)
            except Exception:
                rec = None
        pub = None
        pstore = getattr(self, "publication_store", None)
        if pstore is not None:
            try:
                pub = pstore.get(key)
            except Exception:
                pub = None
        modes_fn = getattr(self, "_active_frame_record_modes", None)
        try:
            m1, m2 = modes_fn(None, None) if callable(modes_fn) else (None, None)
        except Exception:
            m1, m2 = None, None
        return rec, pub, m1, m2

    def _snapshot_data(self, idxs, *, allow_blocking_read=None):
        """Return a small {idx: (frame_1d, frame_2d_dict)} dict for
        the requested ``idxs``, sampled atomically under
        ``self.data_lock``.

        M3 helper: callers that need to iterate over a set of
        frames' data (e.g. for averaging) should take the snapshot
        once and then process it without holding the lock.  This
        keeps the lock window short (one dict-comprehension's worth)
        and avoids the "dictionary changed size during iteration"
        race that the wrangler thread can otherwise trigger.

        Missing frames are silently omitted from the result — the
        caller is expected to handle partial data anyway.
        """
        snapshot = {}
        missing = []
        scan_store_active = getattr(self, "viewer_mode", None) is None
        store_first_lookup = getattr(
            self, "_store_first_publication_for_display", None)
        use_store_first = (
            scan_store_active
            and getattr(self, "store_first_frame_view", None) is not None
            and store_first_lookup is not None
        )
        publication_lookup = getattr(
            self, "_publication_from_store_for_display", None)
        block_check = getattr(self, "_display_hydration_should_block", None)
        should_block = (
            block_check(allow_blocking_read)
            if callable(block_check)
            else bool(allow_blocking_read)
        )
        for idx in idxs:
            key = int(idx)
            publication = None
            if use_store_first:
                publication = store_first_lookup(
                    key, allow_blocking_read=should_block)
            elif publication_lookup is not None:
                publication = publication_lookup(
                    key, allow_blocking_read=allow_blocking_read)
            if publication is None:
                missing.append(key)
                continue
            snapshot[key] = self._publication_legacy_parts(publication)
        if missing and not scan_store_active:
            with self.data_lock:
                for key in missing:
                    snapshot[key] = (
                        self.viewer_rows_1d.get(key),
                        self.viewer_rows_2d.get(key, {}),
                    )
        return snapshot

    def _hydrate_frame_from_disk(self, idx, *, allow_blocking_read=True):
        """Lazy-load a frame from the scan for a ``viewer_rows_2d`` cache miss.

        ``viewer_rows_2d`` is a bounded window (``FixSizeOrderedDict(max=20)``), so a
        cross-frame 2D selection larger than the window would otherwise drop the
        evicted frames silently.  This pulls the missing frame from the in-memory
        series (``scan.frames`` — itself a 64-deep cache that lazy-reloads from
        the ``.nxs``), giving its ``int_2d`` (cake, read from the stacked
        ``integrated_2d``) and ``thumbnail``.  ``map_raw`` is NOT in the ``.nxs``;
        callers that need it call ``_lazy_load_raw`` on the returned frame.
        Returns the ``LiveFrame`` or ``None`` (never raises — a missing/corrupt
        frame is just excluded from the selection).
        """
        scan = getattr(self, 'scan', None)
        frames = getattr(scan, 'frames', None)
        if frames is None:
            return None
        # While a run is active the wrangler is writing the .nxs.  Opening it
        # here (frames[idx] -> LiveFrameSeries.__getitem__ -> catch_h5py_file,
        # which retries the h5py open 100x x 50ms under the writer's file_lock)
        # would block the GUI thread for ~5s per evicted frame -> multi-minute
        # freeze over a long scan.  Serve a cache miss from the writer's
        # already-resident in-memory frames only -- a lock-free single-key dict
        # read (atomic under the GIL; never marks _persisted, so persist-before-
        # evict is untouched) -- and skip anything not resident until the run
        # goes idle.  The full disk hydration below runs only when idle (post-run
        # reload / whole-scan Set-Bkg on a finished file), where the writer is
        # not contending and catch_h5py_file opens cleanly.
        if getattr(self, '_processing_active', False):
            in_mem = getattr(frames, '_in_memory', None)
            return in_mem.get(int(idx)) if isinstance(in_mem, dict) else None
        if not allow_blocking_read:
            # D2 (greenfield Phase 3): the GUI render thread must NOT open the
            # .nxs here — that synchronous catch_h5py_file open is the ~5 s
            # scroll-back / Set-Bkg freeze.  Serve only a resident in-memory
            # frame; an evicted frame is rehydrated OFF-thread by the
            # FrameHydrationWorker (which calls this with the default
            # allow_blocking_read=True), and the panel repaints on completion.
            in_mem = getattr(frames, '_in_memory', None)
            return in_mem.get(int(idx)) if isinstance(in_mem, dict) else None
        try:
            if int(idx) not in frames.index:
                return None
            return frames[int(idx)]
        except (KeyError, RuntimeError, OSError, ValueError, TypeError):
            logger.debug("hydrate frame %s from disk failed", idx, exc_info=True)
            return None

    @staticmethod
    def _scan_file_lock(owner, scan=None):
        """Return the writer-coordinating lock for display-layer .nxs reads."""
        scan = scan if scan is not None else getattr(owner, "scan", None)
        frames = getattr(scan, "frames", None)
        return (
            getattr(owner, "file_lock", None)
            or getattr(scan, "file_lock", None)
            or getattr(frames, "file_lock", None)
        )

    @contextmanager
    def _locked_scan_read(self, scan=None):
        """Context manager for display/hydration reads of live-writable .nxs files.

        Writers hold the same lock around `_save_to_nexus`; every new display
        reader that opens the processed scan file should enter here first.
        """
        lock = DisplayDataMixin._scan_file_lock(self, scan)
        ctx = lock if lock is not None else nullcontext()
        with ctx:
            yield

    def _rehydrate_publication(self, label):
        """D2 hydrator for the shared :class:`PublicationStore` (greenfield
        Phase 3): read an evicted frame from ``scan.frames`` / the ``.nxs`` and
        build a heavy :class:`FramePublication` (cake + full raw).

        Registered via ``store.set_hydrator`` and invoked by
        ``store.get_or_hydrate`` — from the BACKGROUND
        :class:`FrameHydrationWorker` thread, NEVER the GUI thread (the ``.nxs``
        open is the ~5 s scroll-back / Set-Bkg freeze this whole machinery
        exists to move off the GUI thread).  Returns ``None`` on a miss (the
        store keeps whatever lighter/thumbnail-tier publication it had).  During
        an active run ``_hydrate_frame_from_disk`` serves only the writer's
        resident in-memory frames (lock-free), so this never contends with the
        live writer for the file.
        """
        try:
            lf = self._hydrate_frame_from_disk(int(label))
        except Exception:
            logger.debug("rehydrate: disk read failed for %s", label,
                         exc_info=True)
            return None
        if lf is None:
            return None
        if getattr(lf, 'map_raw', None) is None:
            try:
                lf._lazy_load_raw()
            except Exception:
                logger.debug("rehydrate: lazy raw failed for %s", label,
                             exc_info=True)
        store = getattr(self, 'publication_store', None)
        generation = store.generation if store is not None else 0
        try:
            from xdart.modules.frame_publication import publication_from_live_frame
            # Step 6: key the rehydrated record under the real GI mode (same as
            # the live/reintegrate upsert sites) so an evicted-then-rehydrated
            # frame's record is consistent with the rest.  .view is unaffected.
            _scan = getattr(self, "scan", None)
            _display_gi = getattr(self, "_display_gi_enabled", None)
            _is_gi = (
                bool(_display_gi(_scan))
                if callable(_display_gi) and _scan is not None
                else bool(getattr(_scan, "gi", False))
            )
            _args_getter = getattr(self, "_display_bai_args", None)
            _a1 = (
                _args_getter(_scan, "1d")
                if callable(_args_getter)
                else getattr(_scan, "bai_1d_args", {}) or {}
            )
            _a2 = (
                _args_getter(_scan, "2d")
                if callable(_args_getter)
                else getattr(_scan, "bai_2d_args", {}) or {}
            )
            return publication_from_live_frame(
                lf, generation=generation, include_raw=True,
                active_mode_1d=(
                    _a1.get("gi_mode_1d", "q_total")
                    if _is_gi else None),
                active_mode_2d=(
                    _a2.get("gi_mode_2d", "qip_qoop")
                    if _is_gi else None),
            )
        except Exception:
            logger.debug("rehydrate: publication build failed for %s", label,
                         exc_info=True)
            return None

    def _rehydrate_publications_1d(self, labels):
        """Batch hydrate selected 1D rows without raw/cake materialization."""
        labels = tuple(dict.fromkeys(int(label) for label in labels))
        if not labels:
            return ()
        scan = getattr(self, "scan", None)
        scan_file = getattr(scan, "data_file", None)
        if not scan_file:
            return ()
        chunk_size = int(
            getattr(self, "_bulk_1d_read_chunk", _BULK_1D_READ_CHUNK) or
            _BULK_1D_READ_CHUNK
        )
        try:
            from xrd_tools.io import get_1d
        except Exception:
            logger.debug("batch 1D rehydrate import failed", exc_info=True)
            return ()
        results = []
        for chunk in _chunked(labels, chunk_size):
            try:
                with DisplayDataMixin._locked_scan_read(self, scan):
                    results.append(get_1d(scan_file, frame=chunk))
            except Exception:
                logger.debug("batch 1D rehydrate failed for %s", chunk,
                             exc_info=True)
        if not results:
            return ()

        store = getattr(self, "publication_store", None)
        generation = store.generation if store is not None else 0
        mode_1d = DEFAULT_MODE_KEY
        display_gi = getattr(self, "_display_gi_enabled", None)
        if (
            bool(display_gi(scan))
            if callable(display_gi) and scan is not None
            else bool(getattr(scan, "gi", False))
        ):
            args_getter = getattr(self, "_display_bai_args", None)
            args = (
                args_getter(scan, "1d")
                if callable(args_getter)
                else getattr(scan, "bai_1d_args", {}) or {}
            )
            mode_1d = args.get("gi_mode_1d", "q_total")

        publications = []
        for result in results:
            frames = np.asarray(result.frames).ravel()
            intensity = np.asarray(result.intensity)
            if intensity.ndim == 1:
                intensity = intensity[np.newaxis, :]
            sigma = result.sigma
            if sigma is not None:
                sigma = np.asarray(sigma)
                if sigma.ndim == 1:
                    sigma = sigma[np.newaxis, :]
            axis_1d = axis_from_unit(result.q_unit,
                                     np.asarray(result.q, dtype=float))
            for row, frame in enumerate(frames):
                label = int(frame)
                existing = store.get(label) if store is not None else None
                base_view = getattr(existing, "view", None)
                metadata_raw = dict(
                    getattr(existing, "metadata_raw", None)
                    or getattr(base_view, "metadata_raw", None)
                    or {}
                )
                metadata_numeric = dict(
                    getattr(existing, "metadata_numeric", None)
                    or getattr(base_view, "metadata_numeric", None)
                    or numeric_metadata(metadata_raw)
                )
                view = FrameView(
                    label=label,
                    axis_1d=axis_1d,
                    intensity_1d=intensity[row],
                    sigma_1d=(None if sigma is None else sigma[row]),
                    thumbnail=getattr(base_view, "thumbnail", None),
                    mask_baked=bool(getattr(base_view, "mask_baked", False)),
                    metadata_raw=metadata_raw,
                    metadata_numeric=metadata_numeric,
                    incident_angle=getattr(base_view, "incident_angle", None),
                    geometry=getattr(base_view, "geometry", None),
                    source_path=getattr(base_view, "source_path", None),
                    source_frame_index=getattr(base_view, "source_frame_index", None),
                    extra=getattr(base_view, "extra", {}),
                )
                record = FrameRecord.from_view(view, mode_1d=mode_1d)
                publications.append(
                    publication_from_frame_view(
                        view,
                        record=record,
                        generation=generation,
                        source_identity=(
                            getattr(existing, "source_identity", "")
                            or str(scan_file)
                        ),
                        raw_status="1d-only",
                        validate=False,
                    )
                )
        return tuple(publications)

    # ── Raw 2D data access ────────────────────────────────────────

    def get_frames_map_raw(self, idxs=None, *, prefer_thumbnail=False,
                           return_source=False, require_all=False,
                           allow_blocking_read=None):
        """Return 2D frame data for multiple frames (averaged).

        Falls back to the stored thumbnail when full-resolution raw data
        is not available (e.g. when loading from NeXus files that only
        store integration results + thumbnails).

        ``allow_blocking_read=True`` forces a synchronous disk read for evicted
        frames — for one-shot user aggregations (e.g. Set-Bkg over the whole
        scan) that must not silently miss an evicted frame and fall back to
        background=0.  ``None`` (default) keeps the per-render async-gated
        behaviour: non-blocking in the live app, blocking headless.

        M3: takes a single snapshot of the requested idxs under
        ``data_lock`` and then iterates the snapshot lock-free, so
        a concurrent writer can keep streaming new frames without
        racing the iteration.
        """
        if idxs is None:
            idxs = self.idxs_2d
        idxs = list(idxs)

        snapshot = self._snapshot_data(idxs, allow_blocking_read=allow_blocking_read)

        intensity, ctr = 0., 0
        sources = set()
        sanitize = getattr(
            self, '_sanitize_display_image',
            DisplayDataMixin._sanitize_display_image,
        )
        # uint16-65535 saturation masking is opt-in via the wrangler's
        # "Mask saturated" toggle (carried onto the scan); default ON.
        _mask_sat = bool(getattr(getattr(self, 'scan', None),
                                 'mask_sentinel', True))
        for nn, idx in enumerate(idxs):
            frame_1d, frame_2d = snapshot.get(int(idx), (None, {}))
            raw = frame_2d.get('map_raw')
            # Cache the full-resolution detector shape from any resident raw so
            # the thumbnail render path can map the flat detector gap mask into
            # thumbnail coordinates.  update_image re-masks gaps when a frame's
            # thumbnail was generated without the bake (e.g. the last frame
            # persisted at end-of-scan), where it would otherwise show the
            # 0-valued module gaps as dark instead of NaN.
            if raw is not None and getattr(raw, 'ndim', 0) == 2:
                self._raw_full_shape = tuple(raw.shape)
            bg = frame_2d.get('bg_raw', 0)
            if bg is None:                  # LRU eviction nulls bg_raw
                bg = 0
            # Try thumbnail from viewer_rows_2d, then fall back to viewer_rows_1d
            thumb = frame_2d.get('thumbnail')
            if thumb is None and frame_1d is not None:
                thumb = getattr(frame_1d, 'thumbnail', None)
            # Hydrate from disk when full-res raw is missing.  Two cases:
            # (a) total cache miss (no thumbnail either) -- the original
            #     Set-Bkg-over-the-whole-scan path; any selection size.
            # (b) thumbnail present but raw missing, SINGLE-frame selection
            #     without an explicit thumbnail preference: the load worker
            #     publishes the thumbnail preview first and a raw
            #     replace-chunk second -- if that second chunk failed
            #     (source unresolvable on this machine) or was dropped
            #     (generation gate), the panel previously stranded on the
            #     thumbnail forever.  Multi-frame averages keep thumbnails
            #     (no N x 18 MB loads); a per-index negative cache (cleared
            #     with the display caches) keeps unresolvable sources from
            #     re-attempting a file open on every render.
            _hydrate = getattr(self, '_hydrate_frame_from_disk', None)
            _failed = getattr(self, '_raw_resolve_failed', None)
            _want_hydrate = (
                raw is None
                and _hydrate is not None
                and not (_failed and int(idx) in _failed)
                and (thumb is None
                     or (not prefer_thumbnail and len(idxs) == 1))
            )
            if _want_hydrate:
                # D2: when async hydration is enabled (live app), never block the
                # GUI thread on the .nxs open — serve a resident frame only and
                # queue the evicted one for the background worker (the panel keeps
                # its thumbnail and repaints on completion).  Headless/sync: full
                # blocking read as before.
                _async = getattr(self, '_async_hydration_enabled', False)
                _block = (not _async) if allow_blocking_read is None \
                    else bool(allow_blocking_read)
                lf = _hydrate(int(idx), allow_blocking_read=_block)
                if lf is None and not _block and _async:
                    self._request_frame_hydration(int(idx))
                if lf is not None:
                    if getattr(lf, 'map_raw', None) is None:
                        try:
                            lf._lazy_load_raw()
                        except Exception:
                            logger.debug("lazy raw reload failed for %s", idx,
                                         exc_info=True)
                    raw = getattr(lf, 'map_raw', None)
                    if raw is not None and getattr(raw, 'ndim', 0) == 2:
                        self._raw_full_shape = tuple(raw.shape)
                    # free_raw() nulls bg_raw and _lazy_load_raw restores
                    # only map_raw -- the attribute EXISTS with value None,
                    # so the getattr default never applies; raw - None
                    # raised TypeError on the GUI thread (delta review).
                    bg = getattr(lf, 'bg_raw', 0)
                    if bg is None:
                        bg = 0
                    if thumb is None:
                        thumb = getattr(lf, 'thumbnail', None)
                    if frame_1d is None:
                        frame_1d = lf
                    # Don't leave the lazily-loaded ~18 MB raw pinned on the
                    # shared in-memory frame (it would re-inflate the 64-deep
                    # cache and defeat the wrangler's free_raw discipline).  The
                    # local ``raw`` ref keeps it alive for the accumulate below;
                    # free_raw is a no-op when the source isn't reloadable.
                    lf.free_raw()
                if raw is None:
                    # Mark only GENUINE resolve failures.  During a run the
                    # hydrate helper serves in-memory frames only (a miss is
                    # transient), and an idx not yet in the scan index is a
                    # load race -- marking those suppressed the post-run
                    # self-heal permanently (delta review).
                    transient = getattr(self, '_processing_active', False)
                    if not transient:
                        try:
                            transient = int(idx) not in getattr(
                                self.scan.frames, 'index', ())
                        except Exception:
                            transient = True
                    if not transient:
                        if _failed is None:
                            _failed = self._raw_resolve_failed = set()
                        _failed.add(int(idx))
            # Stage 1: the raw-vs-thumbnail-vs-none decision is the pure
            # ``choose_raw_source`` (unit-tested headlessly).  want_raw is
            # always True here — this path never refuses full raw data.
            src = choose_raw_source(
                raw is not None, thumb is not None,
                prefer_thumbnail=prefer_thumbnail, want_raw=True,
            )
            if src is RawSource.THUMBNAIL and prefer_thumbnail:
                # Honour an explicit thumbnail preference: feed the thumbnail
                # through the raw path with no background subtraction (its
                # mask is already baked in).
                raw = thumb
                bg = 0
            source = src.value if src is not RawSource.NONE else None
            # F1: was `for kk in range(3): try: ...; break; except
            # ValueError: time.sleep(0.5)`.  The retry/sleep pattern
            # was running on the Qt thread — visible UI freeze on
            # any single broken frame.  The ValueError originated from
            # shape mismatches during early-load races; we now log
            # them at debug and move on (the GUI re-fires update
            # signals on its own when the wrangler finishes more
            # frames, so a missed average will be recomputed next
            # cycle).
            try:
                scan_info = frame_1d.scan_info if frame_1d is not None else {}
                if raw is not None:
                    raw_data = sanitize(raw, mask_saturation=_mask_sat)
                    intensity += self.normalize(raw_data - bg, scan_info)
                    ctr += 1
                    sources.add(source or 'raw')
                elif thumb is not None:
                    # Use thumbnail as fallback when raw isn't stored
                    thumb_data = sanitize(thumb, mask_saturation=_mask_sat)
                    intensity += self.normalize(
                        thumb_data, scan_info)
                    ctr += 1
                    sources.add('thumbnail')
            except (ValueError, TypeError) as e:
                logger.debug(
                    "get_frames_map_raw skipped frame %s due to shape "
                    "mismatch: %s", idx, e,
                )

        if require_all and ctr != len(idxs):
            if return_source:
                return None, None
            return None

        if ctr > 0:
            intensity /= ctr
        else:
            if return_source:
                return None, None
            return None

        if len(sources) == 1:
            source = next(iter(sources))
        else:
            source = 'mixed'
        data = np.asarray(intensity, dtype=float)
        if return_source:
            return data, source
        return data

    # G2: get_scan_map_raw was deleted.  It read scan.overall_raw,
    # an in-memory accumulator that doesn't survive v2 reload (the
    # loader doesn't repopulate it) and goes stale under R1's
    # replace-frames save.  The Overall view in update_image now
    # aggregates via get_frames_map_raw(list(scan.frames.index)).

    # ── 2D integration data access ────────────────────────────────

    def get_frames_int_2d(self, idxs=None, *, require_all=False,
                          allow_blocking_read=None):
        """Return 2D frame data for multiple frames (averaged).

        Mirrors :meth:`get_frames_map_raw` / :meth:`get_frames_int_1d`:
        accumulates per-frame normalized intensity from ``viewer_rows_2d`` and
        averages on the fly. No external state required — always reflects
        the current selection in ``viewer_rows_2d``.

        Returns ``(intensity, xdata, ydata)`` or ``(None, None, None)``
        if nothing usable is loaded.

        M3: uses ``_snapshot_data`` for a stable view of the
        requested idxs; concurrent writes to ``viewer_rows_1d``/``viewer_rows_2d``
        no longer race this iteration.
        """
        if idxs is None:
            idxs = self.idxs_2d
        idxs = list(idxs)

        if not idxs:
            return None, None, None

        snapshot = self._snapshot_data(
            idxs, allow_blocking_read=allow_blocking_read)
        should_block = self._display_hydration_should_block(
            allow_blocking_read)

        intensity = None
        xdata = ydata = None
        ref_radial = ref_azimuthal = None
        ref_unit = ref_azimuthal_unit = ""
        ref_frame = None
        ctr = 0
        for idx in idxs:
            frame_1d, frame_2d = snapshot.get(int(idx), (None, None))
            if frame_2d is None or frame_2d.get('int_2d') is None:
                # viewer_rows_2d cache miss (frame outside the bounded 20-deep window):
                # hydrate the cake from the on-disk integrated_2d stack so a
                # selection larger than the cache averages ALL selected frames,
                # not just the cached subset (the silent-partial bug).
                lf = self._hydrate_frame_from_disk(
                    int(idx), allow_blocking_read=should_block)
                if lf is None or getattr(lf, 'int_2d', None) is None:
                    if not should_block:
                        self._request_missing_publication(int(idx))
                    continue
                frame_1d = lf
                frame_2d = {
                    'int_2d': lf.int_2d,
                    'gi_2d': getattr(lf, 'gi_2d', {}) or {},
                }
            ir2d = frame_2d['int_2d']
            _gi2d = frame_2d.get('gi_2d', {})
            try:
                _i = self.get_int_2d(ir2d, frame_1d, gi_2d=_gi2d)
            except (ValueError, AttributeError, TypeError):
                continue
            if _i.ndim != 2:
                continue
            radial = np.asarray(getattr(ir2d, 'radial', None), dtype=float)
            azimuthal = np.asarray(getattr(ir2d, 'azimuthal', None), dtype=float)
            if intensity is None:
                intensity = np.asarray(_i, dtype=float)
                ref_radial, ref_azimuthal = radial, azimuthal
                ref_unit = str(getattr(ir2d, 'unit', '') or '')
                ref_azimuthal_unit = str(getattr(ir2d, 'azimuthal_unit', '') or '')
                ref_frame = frame_1d
                try:
                    xdata, ydata = self.get_xydata(ir2d, gi_2d=_gi2d, frame=frame_1d)
                except (ValueError, AttributeError, TypeError):
                    xdata, ydata = radial, azimuthal
            else:
                # Axis-identity guard: the reducer sums element-wise, so every
                # frame must share ONE q/chi grid.  Compare the cake's own
                # radial/azimuthal (not the display-converted axes).  The writer
                # enforces within-scan uniform 2D axes; this makes that explicit
                # and future-proofs cross-source sums (stitch/RSM) — a frame on
                # a different grid is excluded, never silently misaligned.
                if not (_axes_close(radial, ref_radial)
                        and _axes_close(azimuthal, ref_azimuthal)):
                    logger.warning(
                        "get_frames_int_2d: frame %s cake grid differs from the "
                        "selection grid; excluded from the average.", idx)
                    continue
                try:
                    intensity += _i
                except (ValueError, AttributeError, TypeError):
                    continue
            ctr += 1

        if require_all and ctr != len(idxs):
            return None, None, None

        if intensity is None or ctr == 0:
            return None, None, None

        intensity /= ctr
        # get_frames_int_2d feeds ONLY Set-Bkg (display_frame_widget.setBkg); the
        # cake DISPLAY draws from the payload and resamples Q<->2theta there.  So
        # return the NATIVE-grid intensity: resampling the background onto the
        # display grid (as 9034302 briefly did here) makes it bin-MISALIGNED with
        # the native cakes it is later subtracted from whenever a 2D background is
        # captured under a non-default imageUnit (the cake DISPLAY resamples the
        # difference afterward, so doing it here too is both wrong and redundant).
        return intensity, xdata, ydata

    # G2: get_scan_int_2d was deleted.  It read scan.bai_2d, an
    # in-memory accumulator that doesn't survive v2 reload.  The
    # Overall view in update_binned now uses
    # get_frames_int_2d(list(scan.frames.index)).  The comment
    # at the call site (display_frame_widget.update_binned) already
    # noted this path returned 1×1 zeros for NeXus files — so it's
    # been functionally dead since v2 landed.

    def get_int_2d(self, int_2d, frame_1d=None, normalize=True, gi_2d=None):
        """Returns the appropriate 2D data depending on the chosen axes.
        In GI mode, int_2d already holds the selected mode's data.
        """
        if int_2d is None:
            return np.zeros((1, 1))
        # int_2d is always the correct result (GI or standard)
        intensity_2d = np.asarray(int_2d.intensity)
        intensity = (
            intensity_2d.astype(float, copy=True)
            if normalize
            else intensity_2d
        )

        if normalize:
            if frame_1d is not None:
                intensity = self.normalize(intensity, frame_1d.scan_info)
            else:
                norm_fac = len(self.scan.frames.index)
                if self.normChannel:
                    # scan_data may now carry non-numeric columns (N2); a
                    # non-numeric norm channel degrades to no normalization
                    # rather than crashing on a string ``.sum()``.
                    try:
                        norm = float(self.scan.scan_data[self.normChannel].sum())
                    except (TypeError, ValueError):
                        norm = 0.0
                    if norm > 0:
                        norm_fac = norm
                intensity /= norm_fac

        return intensity

    # ── 1D integration data access ────────────────────────────────

    def get_frames_int_1d(self, idxs=None, rv='all', *, require_all=False,
                          allow_blocking_read=None):
        """Return 1D data for multiple frames"""
        if idxs is None:
            idxs = self.idxs_1d
        idxs = list(idxs)
        snapshot = self._snapshot_data(
            idxs, allow_blocking_read=allow_blocking_read)
        should_block = self._display_hydration_should_block(
            allow_blocking_read)

        # Collect rows then stack once.  The previous code re-allocated a
        # growing array with np.vstack on every iteration — O(N^2) over the
        # frames of a Waterfall/Overlay/Sum/Average; one stack at the end is
        # O(N).
        xdata = None
        ys: list = []
        for idx in idxs:
            frame_1d, frame_2d = snapshot.get(int(idx), (None, None))
            if frame_1d is None or getattr(frame_1d, 'int_1d', None) is None:
                # Publication-store miss (or no store on a lightweight test
                # host): hydrate from disk so a 1D sum/average / Set-Bkg over
                # the whole scan covers ALL selected frames.
                lf = self._hydrate_frame_from_disk(
                    int(idx), allow_blocking_read=should_block)
                if lf is None or getattr(lf, 'int_1d', None) is None:
                    if not should_block:
                        self._request_missing_publication(int(idx), purpose="1d")
                    continue
                frame_1d = lf
                if frame_2d is None:
                    frame_2d = {
                        'int_2d': getattr(lf, 'int_2d', None),
                        'gi_2d': getattr(lf, 'gi_2d', {}) or {},
                    }
            x, y = self.get_int_1d(frame_1d, frame_2d, idx)
            if x is None or y is None:
                continue
            if xdata is None:
                xdata = x
            ys.append(y)

        if require_all and len(ys) != len(idxs):
            return None, None

        if not ys:
            return None, None

        ydata = ys[0] if len(ys) == 1 else np.vstack(ys)

        if ydata.ndim == 2:
            if rv == 'average':
                ydata = np.nanmean(ydata, 0)
            elif rv == 'sum':
                ydata = np.nansum(ydata, 0)

        return ydata, xdata

    def get_int_1d(self, frame, frame_2d, idx):
        """Returns 1D integrated data for frame.

        Uses ``self._plot_axis_info`` to determine whether the selected
        plotUnit axis comes from the 1D integration (direct readout) or
        the 2D integration (requires slicing/projection from the 2D map).
        When the axis is 2D-derived *and* slicing is enabled, only the
        selected range of the orthogonal axis is averaged.
        """
        _plot_idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[_plot_idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= _plot_idx < len(self._plot_axis_info)
                else {'source': '1d', 'slice_axis': None, 'axis': None})

        # Pure 2D axes always need 2D data; hybrid (1d_2d) only when slicing
        _needs_2d = (info['source'] == '2d') or \
                    (info['source'] == '1d_2d' and self.ui.slice.isChecked())

        # --- Fast path: pure 1D readout (no 2D data needed) ---
        if not _needs_2d:
            int_1d = frame.int_1d
            if int_1d is None:
                return None, None
            intensity = int_1d.intensity
            ydata = self.normalize(intensity, frame.scan_info)
            xdata = self.get_xdata(frame)
            return xdata, ydata

        # --- 2D path: project from 2D map ---
        if frame_2d is None or frame_2d.get('int_2d') is None:
            return None, None

        intensity = self.get_int_2d(frame_2d['int_2d'], frame, normalize=False,
                                    gi_2d=frame_2d.get('gi_2d', {}))
        if intensity.ndim < 2:
            return None, None

        _i2d = frame_2d['int_2d']
        radial = _i2d.radial if _i2d is not None else np.array([])
        azimuthal = _i2d.azimuthal if _i2d is not None else np.array([])

        # Determine which 2D axis is the "display" axis and which is
        # the "slice" axis.
        # IntegrationResult2D.intensity shape is [radial, azimuthal].
        axis_type = info.get('axis', 'radial')

        if axis_type == 'radial':
            # Display along radial, slice along azimuthal
            xdata = radial
            # Convert the cake's radial axis to the SELECTED plotUnit (Q<->2θ) so a
            # 2D-DERIVED radial entry returns the right unit regardless of the
            # cake's integration unit — e.g. the chi-integration mode's "2θ" cake
            # entry must return degrees even when the cake was integrated in Q (and
            # the standard-mode sliced "2θ" entry likewise).  Standard mode only;
            # GI reciprocal axes are verbatim (convert_2d_radial is a no-op for GI
            # and when the wavelength is unknown / the units already match).
            if not getattr(self.scan, 'gi', False):
                from .display_logic import convert_2d_radial
                from .display_constants import AA_inv, Th
                plot_label = self.ui.plotUnit.currentText()
                xdata = convert_2d_radial(
                    xdata,
                    data_unit=getattr(_i2d, 'unit', 'q_A^-1'),
                    want_tth=(Th in plot_label),
                    want_q=(AA_inv in plot_label),
                    wavelength_m=self._get_wavelength(frame),
                )
            slice_data = azimuthal
            # mean over azimuthal (axis 1) → 1D along radial
            reduce_axis = 1
        elif axis_type == 'azimuthal':
            # Display along azimuthal, slice along radial
            xdata = azimuthal
            slice_data = radial
            # mean over radial (axis 0) → 1D along azimuthal
            reduce_axis = 0
        else:
            # Fallback for legacy standard-mode paths
            xdata = radial
            slice_data = azimuthal
            reduce_axis = 1

        # Apply slice range if enabled
        _inds = np.s_[:]
        if self.ui.slice.isChecked():
            center = self.ui.slice_center.value()
            width = self.ui.slice_width.value()
            _range = [center - width, center + width]
            _inds = (_range[0] <= slice_data) & (slice_data <= _range[1])

        # nanmean_slice never warns "Mean of empty slice" on an all-NaN column
        # (GI empty bins); a 0-bin slice -> None -> keep the array contract with an
        # all-NaN curve of the surviving-axis length.
        from .display_logic import nanmean_slice
        if reduce_axis == 0:
            # Reducing over radial (axis 0): _inds filters radial rows
            ydata = nanmean_slice(intensity[_inds, :], 0)
            if ydata is None:
                ydata = np.full(intensity.shape[1], np.nan)
        else:
            # Reducing over azimuthal (axis 1): _inds filters azimuthal cols
            ydata = nanmean_slice(intensity[:, _inds], 1)
            if ydata is None:
                ydata = np.full(intensity.shape[0], np.nan)

        self.show_slice_overlay()

        ydata = self.normalize(ydata, frame.scan_info)
        return xdata, ydata

    # ── Axis data helpers ─────────────────────────────────────────

    def get_xydata(self, int_2d, gi_2d=None, frame=None):
        """Reads the 2D unit box and returns the appropriate radial / azimuthal
        axes, converting Q ↔ 2θ on the fly when the selected ``imageUnit``
        differs from the integration unit (mirrors :meth:`get_xdata` for the
        1D plot — without this the cake's x-axis stayed in Q while only the
        label switched to 2θ).

        In GI mode ``int_2d`` already holds the selected GI-mode result
        (qz/qxy or q/χ), so the axes are returned as-is — there's no
        Q↔2θ toggle there (the imageUnit combo is fixed/disabled in GI).

        args:
            int_2d: IntegrationResult2D, primary integration result
            gi_2d: dict of IntegrationResult2D for GI modes (unused, kept
                   for API compatibility)
            frame: optional LiveFrame for the wavelength lookup

        returns:
            xdata, ydata: numpy arrays for radial and azimuthal axes.
        """
        if int_2d is None:
            return np.array([]), np.array([])
        radial = np.asarray(int_2d.radial, dtype=float)
        azimuthal = int_2d.azimuthal
        # Return GI reciprocal-space axes verbatim — no Q↔2θ conversion.
        # Honour the result's *units* (not just the live ``scan.gi`` flag):
        # a reloaded qip/qoop cake whose scan.gi wasn't restored would
        # otherwise be run through the q→2θ arcsin path (out-of-range qip →
        # collapsed/blank cake).  See display_logic.is_gi_2d_units.
        from .display_logic import is_gi_2d_units
        if getattr(self.scan, 'gi', False) or is_gi_2d_units(
                getattr(int_2d, 'unit', ''),
                getattr(int_2d, 'azimuthal_unit', '')):
            return radial, azimuthal

        from .display_constants import AA_inv, Th
        from .display_logic import convert_2d_radial

        image_label = self.ui.imageUnit.currentText()
        radial = convert_2d_radial(
            radial,
            data_unit=getattr(int_2d, 'unit', 'q_A^-1'),
            want_tth=(Th in image_label),       # imageUnit label names 2θ
            want_q=(AA_inv in image_label),      # imageUnit label names Q (Å⁻¹)
            wavelength_m=self._get_wavelength(frame),
        )
        return radial, azimuthal

    def get_xdata(self, frame):
        """Reads the unit box and returns appropriate xdata for 1D plot.

        Handles on-the-fly Q ↔ 2θ conversion when the plotUnit selection
        differs from the integration unit stored in int_1d.

        args:
            frame: LiveFrame copy (viewer_rows_1d entry) holding int_1d and gi_1d

        returns:
            xdata: numpy array, x axis data for plot.
        """
        from .display_constants import AA_inv, Th

        int_1d = getattr(frame, 'int_1d', None)
        if int_1d is None:
            return np.array([])

        radial = int_1d.radial
        plot_label = self.ui.plotUnit.currentText()

        # Determine if conversion is needed by comparing plotUnit label
        # to the stored integration unit
        data_unit = getattr(int_1d, 'unit', 'q_A^-1')
        want_tth = (Th in plot_label)  # plotUnit label contains θ
        have_tth = ('2th' in data_unit)

        if want_tth and not have_tth:
            # Data is in Q, display wants 2θ: convert Q → 2θ
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                arg = np.clip(radial * lam_A / (4 * np.pi), -1, 1)
                return 2 * np.degrees(np.arcsin(arg))
        elif not want_tth and have_tth and (AA_inv in plot_label):
            # Data is in 2θ, display wants Q: convert 2θ → Q
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                return (4 * np.pi / lam_A) * np.sin(np.radians(radial / 2))

        return radial

    def _get_wavelength(self, frame=None):
        """Return the X-ray wavelength in metres.

        Tries several sources in order:
        1. ``frame.integrator.wavelength`` (available during live processing)
        2. ``self.scan.mg_args['wavelength']`` when it is a real value
        3. ``/entry/instrument/source/wavelength_A`` in the HDF5 file

        Returns None if the wavelength cannot be determined.
        """
        # 1. From the frame's integrator (fastest, works during live runs)
        if frame is not None:
            ai = getattr(frame, 'integrator', None)
            wl = getattr(ai, 'wavelength', None) if ai else None
            wl = normalize_wavelength_m(wl, allow_default_sentinel=True)
            if wl is not None:
                # F1: cache a REAL run wavelength (reject the 1e-10 constructor
                # sentinel) for hydrated rows this run; reset each run boundary.
                _real = normalize_wavelength_m(wl)
                if _real is not None:
                    self._run_wavelength_m = _real
                return wl
            poni = getattr(frame, 'poni', None)
            wl = normalize_wavelength_m(
                getattr(poni, 'wavelength', None),
                allow_default_sentinel=True,
            )
            if wl is not None:
                # F1: cache a REAL run wavelength (reject the 1e-10 constructor
                # sentinel) for hydrated rows this run; reset each run boundary.
                _real = normalize_wavelength_m(wl)
                if _real is not None:
                    self._run_wavelength_m = _real
                return wl

        # 1.5 (F1): the run-scoped wavelength, stamped above from the first
        # frame-backed row's integrator THIS run.  Hydrated rows (raw_ref=None)
        # have no frame mid-run, and _persisted_wavelength_m / the HDF5 fallback
        # (below) are unavailable while ``_run_writing`` -> without this the same
        # append batch mixes units (Q for frame-backed rows, unconvertible for
        # hydrated -> the blank-band / stale-combo regression).  Only trusted
        # during a run; reset at each run boundary in ``set_processing_active``.
        if getattr(self, '_run_writing', False):
            run_wl = getattr(self, '_run_wavelength_m', None)
            if run_wl:
                return run_wl

        # 2. From scan.mg_args (loaded when NXS is opened). Reject the
        # historical 1e-10 m constructor sentinel rather than using it for
        # Q↔2θ conversion.
        scan = getattr(self, 'scan', None)
        persisted_wl = normalize_wavelength_m(
            getattr(scan, '_persisted_wavelength_m', None),
            allow_default_sentinel=True,
        )
        if persisted_wl is not None:
            return persisted_wl
        mg_args = getattr(scan, 'mg_args', None)
        wl = mg_args.get('wavelength', None) if isinstance(mg_args, dict) else None
        wl = normalize_wavelength_m(wl)
        if wl is not None:
            return wl

        # 3. Read the writer's actual v2 NeXus wavelength stamp.  This fallback
        # is intentionally cached per scan/file, including a negative result:
        # large Single/Overlay selections can call _get_wavelength once per
        # rendered frame, and opening the same .nxs hundreds of times on the GUI
        # thread is enough to beachball the app.
        data_file = getattr(scan, 'data_file', None)
        if not data_file:
            return None
        if getattr(self, '_run_writing', False):
            return None
        key = (id(scan), os.fspath(data_file))
        if getattr(self, '_wavelength_cache_key', None) == key:
            return getattr(self, '_wavelength_cache_value', None)
        wl = None
        try:
            import h5py
            with h5py.File(data_file, 'r') as f:
                wl = wavelength_angstrom_to_m(
                    f['entry/instrument/source/wavelength_A'][()] # type: ignore
                )
        except Exception:
            logger.debug("Failed to read wavelength from HDF5 instrument/source group in %s", data_file, exc_info=True)

        self._wavelength_cache_key = key
        self._wavelength_cache_value = wl
        return wl

    def _clear_wavelength_cache(self):
        self._wavelength_cache_key = None
        self._wavelength_cache_value = None

    # ── Normalization ─────────────────────────────────────────────

    def normalize(self, int_data, scan_info):
        """Normalize intensity data by the selected normalization channel.

        args:
            int_data: numpy array, intensity data to normalize
            scan_info: dict, metadata containing normalization counters

        returns:
            intensity: numpy array, normalized data
        """
        try:
            intensity = np.asarray(int_data.copy(), dtype=float)
        except AttributeError:
            return np.zeros((10, 10))

        normChannel = self.get_normChannel(scan_data_keys=scan_info.keys())
        if normChannel and (scan_info[normChannel] > 0):
            intensity /= scan_info[normChannel]

        return intensity

    def get_normChannel(self, scan_data_keys=None):
        """Check to see if normalization channel exists in metadata and return name"""
        if scan_data_keys is None:
            scan_data_keys = self.scan.scan_data.columns
        keys = list(scan_data_keys)
        if not keys:
            return None

        key_by_lower = {str(key).lower(): key for key in keys}
        try:
            selected_actual = self.ui.normChannel.currentData()
        except Exception:
            selected_actual = None
        if selected_actual is not None:
            match = key_by_lower.get(str(selected_actual).lower())
            if match is not None:
                return match

        selected = self.ui.normChannel.currentText()
        alias = _norm_alias_key(selected)
        if alias is None:
            return None
        for display, actual in available_norm_channels(keys):
            if _norm_alias_key(display) == alias or _norm_alias_key(actual) == alias:
                return actual
        return None

    def refresh_norm_channels(self):
        """Populate the normalization combo from current scan metadata."""
        combo = getattr(getattr(self, 'ui', None), 'normChannel', None)
        if combo is None:
            return
        try:
            keys = list(self.scan.scan_data.columns)
        except Exception:
            keys = []
        channels = available_norm_channels(keys)
        current = self.get_normChannel(scan_data_keys=keys)
        signature = tuple(channels)
        if signature == getattr(self, '_norm_channel_signature', None):
            return

        try:
            was_blocked = combo.blockSignals(True)
        except Exception:
            was_blocked = None
        try:
            combo.clear()
            def _add_item(label, data):
                try:
                    combo.addItem(label, data)
                except TypeError:
                    combo.addItem(label)
                    try:
                        combo.setItemData(combo.count() - 1, data)
                    except Exception:
                        pass

            _add_item('Norm Channel', None)
            selected_index = 0
            self._norm_channel_map = {}
            for row, (display, actual) in enumerate(channels, start=1):
                _add_item(display, actual)
                self._norm_channel_map[display] = actual
                if current is not None and str(actual).lower() == str(current).lower():
                    selected_index = row
            combo.setCurrentIndex(selected_index)
            self._norm_channel_signature = signature
            # The content-fit width was computed at init from the .ui
            # placeholder; refit for the real counter names so longer ones
            # aren't clipped in the closed combo.
            try:
                self._fit_combo_width(combo, max_w=170)
            except Exception:
                pass
        finally:
            if was_blocked is not None:
                try:
                    combo.blockSignals(was_blocked)
                except Exception:
                    pass

    # ── Colour generation ─────────────────────────────────────────

    def get_colors(self):
        """Generate a list of RGB colour tuples for plot curves."""
        import matplotlib.pyplot as plt

        colors = (1, 1, 1)
        if self.cmap == 'Default':
            colors_tuples = [plt.get_cmap('tab10'), plt.get_cmap('Set3'), plt.get_cmap('tab20b', 5)]
            for nn, color_tuples in enumerate(colors_tuples):
                if nn == 0:
                    colors = np.asarray(color_tuples.colors)
                else:
                    colors = np.vstack((colors, np.asarray(color_tuples.colors)[:, 0:3]))

            colors_tuples = plt.get_cmap('jet')
            more_colors = colors_tuples(np.linspace(0, 1, len(self.frame_names)))
            colors = np.vstack((colors, more_colors[:, 0:3]))

        else:
            try:
                colors_tuples = plt.get_cmap(self.cmap)
            except ValueError:
                colors_tuples = plt.get_cmap('jet', 256)
            colors = colors_tuples(np.linspace(0, 1, len(self.frame_names)))[:, 0:3]

        colors = np.round(colors * [255, 255, 255]).astype(int)
        colors = [tuple(color[:3]) for color in colors]

        return colors

    # ── Stubs for future implementation ───────────────────────────

    def get_profile_chi(self, frame):
        """Extract intensity profile along chi from frame.

        Args:
            frame: LiveFrame object with 2D integration data.

        Returns:
            ndarray: Intensity integrated along chi over the Q range
                     specified by the UI slice controls.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError("get_profile_chi is not yet implemented")

    def get_chi_1d(self, frame):
        """Extract 1D chi profile from frame.

        Args:
            frame: LiveFrame object with 2D integration data.

        Returns:
            ndarray: 1D intensity vs chi extracted from 2D data.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError("get_chi_1d is not yet implemented")

    # ── Save / Export ─────────────────────────────────────────────

    def save_image(self):
        """Saves currently displayed image. Formats are automatically
        grabbed from Qt. Also implements tiff saving.
        """
        import pyqtgraph
        import pyqtgraph.exporters
        from pyqtgraph.Qt import QtWidgets
        from xdart.utils import split_file_name

        QFileDialog = QtWidgets.QFileDialog

        formats = [
            str(f.data(), encoding='utf-8').lower() for f in
            pyqtgraph.Qt.QtGui.QImageReader.supportedImageFormats()
        ]

        ext_filter = "Images ("
        for f in formats:
            ext_filter += "*." + f + " "

        dialog = QFileDialog()
        fname, _ = dialog.getSaveFileName(
            dialog,
            filter=ext_filter,
            caption='Save as...',
            options=QFileDialog.DontUseNativeDialog
        )
        if fname == '':
            return

        # Choose the right widget depending on viewer mode
        if self.viewer_mode == 'image':
            data, rect = self.image_data
            scene = self.image_widget.imageViewBox.scene()
        else:
            data, rect = self.binned_data
            scene = self.binned_widget.imageViewBox.scene()

        exporter = pyqtgraph.exporters.ImageExporter(scene)
        h = exporter.params.param('height').value()
        w = exporter.params.param('width').value()
        if h == 0 or w == 0:
            logger.warning("Cannot export image with zero dimensions (%dx%d)", w, h)
            return
        h_new = 2000
        w_new = int(np.round(w/h * h_new, 0))
        exporter.params.param('height').setValue(h_new)
        exporter.params.param('width').setValue(w_new)
        exporter.export(fname)

        directory, base_name, ext = split_file_name(fname)
        save_fname = os.path.join(directory, base_name)

        # Save as Numpy array
        np.save(f'{save_fname}.npy', data)

        # In image viewer mode, also save a pyFAI-compatible TIFF
        # from the raw detector-frame data (not the transposed display).
        if self.viewer_mode == 'image' and len(self.idxs_2d) > 0:
            try:
                import fabio
                with self.data_lock:
                    _d2 = self.viewer_rows_2d.get(self.idxs_2d[0]) or {}
                raw = np.asarray(_d2.get('map_raw'), dtype=np.float32)
                tif_path = os.path.join(directory, f'{base_name}_npy.tif')
                fabio.tifimage.TifImage(data=raw).write(tif_path)
                logger.info("Saved pyFAI-compatible TIFF: %s", tif_path)
            except Exception:
                logger.exception("Failed to save TIFF for pyFAI")

    def save_1D(self, auto=False):
        """Saves currently displayed data. Currently supports .xye
        and .csv.
        """
        import pyqtgraph
        import pyqtgraph.exporters
        from pyqtgraph.Qt import QtWidgets
        import xdart.utils as ut

        QFileDialog = QtWidgets.QFileDialog

        fname = f'{self.scan.name}'
        if not auto:
            path = QFileDialog.getExistingDirectory(
                self,
                caption="Select Directory to Save Images",
                dir="",
                options=(QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
            )

            inp_dialog = QtWidgets.QInputDialog()
            suffix, ok = inp_dialog.getText(inp_dialog, 'Enter Suffix to be added to File Name', 'Suffix', text='')
            if not ok:
                return
            if suffix != '':
                fname += f'_{suffix}'
        else:
            path = os.path.dirname(self.scan.data_file)
            path = os.path.join(path, self.scan.name)
            Path(path).mkdir(parents=True, exist_ok=True)

        fname = os.path.join(path, fname)

        xdata, ydata = self.plot_data
        # H4: Average / Sum produces ONE combined output.  Pre-H4 the
        # code wrote the combined file AND then fell through to the
        # per-frame loop below — silently producing dozens of extra
        # per-frame .xye files alongside an "average.xye" in the
        # same directory.  Branch cleanly: Average/Sum → combined
        # only; everything else (Overlay, Single, Waterfall) →
        # per-frame files.
        if self.plotMethod in ('Average', 'Sum'):
            if self.plotMethod == 'Average':
                s_ydata = np.nanmean(ydata, 0)
            else:
                s_ydata = np.nansum(ydata, 0)
            xye_fname = f'{fname}.xye'
            ut.write_xye(xye_fname, xdata, s_ydata)
        else:
            idxs = [
                frame.replace(f'{self.scan.name}_', '')
                for frame in self.frame_names
            ]
            for s_ydata, idx in zip(ydata, idxs):
                xye_fname = f'{fname}_{str(idx).zfill(4)}.xye'
                ut.write_xye(xye_fname, xdata, s_ydata)

        if not auto:
            scene = self.plot_viewBox.scene()
            exporter = pyqtgraph.exporters.ImageExporter(scene)
            h = exporter.params.param('height').value()
            w = exporter.params.param('width').value()
            h_new = 600
            w_new = int(np.round(w/h * h_new, 0))
            exporter.params.param('height').setValue(h_new)
            exporter.params.param('width').setValue(w_new)
            exporter.export(fname + '.png')
