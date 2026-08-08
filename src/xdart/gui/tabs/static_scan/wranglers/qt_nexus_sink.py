# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import threading

from xdart.modules.frame_publication import publication_from_live_frame
from xrd_tools.core import DEFAULT_MODE_KEY
from xrd_tools.session import Light1DModeData, Light1DRecord

logger = logging.getLogger(__name__)

class QtFrameObserver:
    def __init__(self, host, scan, publication_store, light_lease, *, generation,
                 active_mode_1d, mask=None, publish_display=True) -> None:
        self._host, self._scan = host, scan
        self._store, self._lease = publication_store, light_lease
        self._generation, self._active_mode_1d = int(generation), active_mode_1d
        self._mask, self._publish_display = mask, bool(publish_display)
        self._registry, self._lock = {}, threading.RLock()

    def register(self, live, attempt_token) -> bool:
        with self._lock:
            label = int(live.idx)
            if label in self._registry:
                return False
            self._registry[label] = (attempt_token, live)
            return True

    def unregister(self, live, attempt_token) -> bool:
        with self._lock:
            label = int(live.idx)
            current = self._registry.get(label)
            if (current is None or current[0] is not attempt_token or current[1] is not live):
                return False
            self._registry.pop(label)
            return True

    @staticmethod
    def _clear_transient_1d(live) -> None:
        live.int_1d = None
        if getattr(live, "gi_1d", None):
            live.gi_1d = {}

    def close(self) -> None:
        with self._lock:
            pending, self._registry = tuple(self._registry.values()), {}
        for _token, live in pending:
            self._clear_transient_1d(live)

    def _hydrate(self, live, event, mode, mode_2d):
        live.scan_info = dict(event.metadata)
        if event.mode_key is None:
            live.int_1d = event.result_1d
            live.int_2d = event.result_2d
        else:
            if event.result_1d is not None:
                live.gi_1d = {**dict(getattr(live, "gi_1d", {}) or {}), mode: event.result_1d}
                live.int_1d = event.result_1d
            if event.result_2d is not None:
                live.gi_2d = {**dict(getattr(live, "gi_2d", {}) or {}), mode_2d: event.result_2d}
                live.int_2d = event.result_2d
        if getattr(live, "thumbnail", None) is None:
            try:
                live.make_thumbnail(global_mask=self._mask)
            except Exception:
                logger.debug("dynamic thumbnail build failed", exc_info=True)

    def on_frame_completed(self, event) -> bool:
        if int(event.generation) != self._generation:
            return False
        label = int(event.frame_index)
        with self._lock:
            registered = self._registry.pop(label, None)
        if registered is None:
            return False
        _attempt_token, live = registered
        mode_key = event.mode_key if isinstance(event.mode_key, tuple) else (DEFAULT_MODE_KEY,) * 2
        mode, mode_2d = (str(value or DEFAULT_MODE_KEY) for value in mode_key)
        if mode != self._active_mode_1d or mode != self._lease.layout.active_mode:
            raise ValueError("dynamic completion mode does not match light-1D layout")
        result = event.result_1d
        if result is None:
            raise ValueError("dynamic GUI publication requires a 1-D result")
        layout = self._lease.layout.modes[0]
        uncertainty = result.sigma if layout.uncertainty is not None else None
        if layout.uncertainty is None and result.sigma is not None:
            raise ValueError("dynamic light-1D uncertainty topology changed")

        try:
            self._hydrate(live, event, mode, mode_2d)
            source_identity = os.path.abspath(str(live.source_file))
            scan_key = str(getattr(self._scan, "name", "") or source_identity)
            publication = publication_from_live_frame(
                live, generation=self._store.generation, source_identity=source_identity,
                include_raw=True, retain_raw_ref=False,
                active_mode_1d=mode, active_mode_2d=mode_2d, scan_key=scan_key)
            light = Light1DRecord(
                row_identity=label, generation=self._lease.generation, active_mode=mode,
                modes={mode: Light1DModeData(result.radial, result.intensity, uncertainty)},
                provenance={"source_identity": source_identity, "scan_key": scan_key})
            self._store.publish_gui_light_1d(publication, light)
        finally:
            self._clear_transient_1d(live)
        if self._publish_display:
            published = getattr(self._host, "_published_frames", None)
            if published is not None:
                published[label] = live
            signal = getattr(self._host, "sigUpdate", None)
            if signal is not None:
                signal.emit(label)
        return True
