> **Provenance:** independent memory-pressure/high-load review (Claude, 2026-07-02) of
> `feature/remediation` pre-tag. Pre-tag fixes dispatched as `codex_tasks/mem1_oom_triad.md`
> (findings [1],[2],[3],[12]); coupled perf work in `codex_tasks/perf_regression_651.md`.
> Post-tag items ([4],[5],[14],[15] + medium/low tail) are ledgered via MEM-1's commit.
> This document is the reference map of the live-mode memory economy.
>
> **Addendum (2026-07-03):** findings [1],[2],[3],[12] fixed in `30eeab4e`
> (MEM-1a-d); RAM-aware heavy window landed in `fb074d6f` (MEM-2).
> [4],[5],[14],[15] remain deferred post-tag (`deferred_ledger.md`). MEM-3
> worker-cap census pending.

# xrd-tools `feature/remediation` — Memory-Pressure & High-Load Review

**Date:** 2026-07-02  
**Reviewer:** Claude Code (memory-load-review workflow, xhigh effort)  
**Scope:** feature/remediation vs master merge-base f2e99a4a (full branch incl. uncommitted)  
**Scenario:** 5000 × 16MB images processed sequentially → 1D traces overlaid into a live waterfall  
**Method:** 4 subsystem retention-maps → 8 finder angles → 2-lens adversarial verify → sweep → rank. 106 agents total; 41 findings survived verification, 20 reported (0 refuted at report time).  
**Baseline commit:** `87e7adc1` + then-uncommitted set. NOTE: the branch has since advanced (`bf4123dd`, `2841fcb6`, `13475970`, `bd5d3bfd`); commit `2841fcb6 Serialize live HDF5 writer and bulk 1D hydration reads` touches the #1 backpressure path, so #1 may be partly mitigated and some line numbers have shifted.

---

## Executive summary

**Does a 5000 × 16 MB → waterfall run stay bounded in memory? Batch mode: yes. Live mode: no — not guaranteed.**

> No — a 5000x16MB waterfall run is NOT guaranteed to stay bounded in live mode. The dominant risk is an unbounded producer-outruns-consumer backlog: the sink writer stores one LiveFrame ref per frame into the uncapped _published_frames map and drains it one-per-queued-GUI-signal, while free_raw() is batch-only and _in_memory eviction never frees raw — so if the GUI falls behind, hundreds-to-thousands of float64-upcast (~64MB) raws pin unbounded, leading to OOM. Even when the GUI keeps pace, live mode holds ~4-8GB of raw across two 64-frame windows (bounded but large vs ~0 in batch). Secondary risks are all CPU/trace-sized (not OOM): O(n^2) waterfall vstack accumulators, full-history scan_data/transpose rebuilds, and a genuine full-2D retention leak where publication-dropped 2D rows become permanently unevictable (multi-GB on GI scans). The viewer-mode select-all raw path is a separate user-driven OOM; batch mode stays bounded, live mode does not without backpressure on _published_frames.

The triad that turns *large-but-bounded* into *OOM* in live mode: **#1 (uncapped `_published_frames`) + #5 (4× float64 upcast) + raw-never-freed-in-live**. Fix the bound on `_published_frames` and free raw on live eviction first. **#2** is the sneakiest — a genuine unbounded leak that does not depend on timing and hits GI scans.

### Severity roll-up

| # | Sev | Category | Verdict | Location |
|---|-----|----------|---------|----------|
| 1 | 🔴 critical | backpressure | CONFIRMED | `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:169` |
| 2 | 🟠 high | leak-retention | CONFIRMED | `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:387` |
| 3 | 🟠 high | leak-retention | CONFIRMED | `src/xdart/gui/tabs/static_scan/viewer_raw_lru.py:65` |
| 4 | 🟠 high | quadratic | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xrd_tools/session/display_logic.py:637` |
| 5 | 🟠 high | copy-bloat | PLAUSIBLE | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:2970` |
| 6 | 🟡 medium | copy-bloat | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_plot.py:1175` |
| 7 | 🟡 medium | quadratic | CONFIRMED | `src/xdart/gui/tabs/static_scan/static_scan_widget.py:4220` |
| 8 | 🟡 medium | quadratic | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/metadata.py:154` |
| 9 | 🟡 medium | quadratic | CONFIRMED | `src/xdart/gui/tabs/static_scan/display_publication.py:903` |
| 10 | 🟡 medium | quadratic | CONFIRMED | `src/xdart/gui/tabs/static_scan/display_plot.py:1128` |
| 11 | 🟡 medium | quadratic | PLAUSIBLE | `src/xdart/gui/tabs/static_scan/display_plot.py:154` |
| 12 | 🟡 medium | correctness | CONFIRMED | `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:3019` |
| 13 | ⚪ low | copy-bloat | PLAUSIBLE | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:1300` |
| 14 | 🟠 high | quadratic | PLAUSIBLE | `src/xdart/gui/tabs/static_scan/display_controllers.py:311` |
| 15 | 🟡 medium | correctness | PLAUSIBLE | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/frame_publication.py:808` |
| 16 | ⚪ low | leak-retention | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xrd_tools/session/frame_record_store.py:378` |
| 17 | ⚪ low | quadratic | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/ewald/scan.py:536` |
| 18 | ⚪ low | quadratic | PLAUSIBLE | `src/xdart/modules/ewald/nexus_writer.py:1500` |
| 19 | ⚪ low | correctness | PLAUSIBLE | `src/xdart/gui/tabs/static_scan/static_scan_widget.py:5636` |
| 20 | ⚪ low | qt-leak | CONFIRMED | `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/h5viewer.py:1032` |

---

## Findings (ranked most-severe first)

### [1] 🔴 CRITICAL · backpressure · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:169`

**Summary:** Live sink publishes one _published_frames[idx] entry + emits sigUpdate per frame with NO producer-side coalescing; the consumer that drains it (update_data) is the same GUI-thread slot that backs up under load, so the map grows one live LiveFrame ref per undrained frame. [same root cause also at: src/xdart/modules/ewald/frame_series.py:499, src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:208, src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:169, /Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/ewald/frame_series.py:499, src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:2531, src/xdart/gui/tabs/static_scan/static_scan_widget.py:4093, src/xdart/modules/ewald/frame_series.py:506]

**Failure @ 5000 frames:** In live/streaming mode, if the writer/reduction thread emits frames faster than the GUI thread drains the queued sigUpdate->update_data slot (very plausible at 5000 fast frames: heavy render legs, list rebuilds, and OS scheduling all steal GUI-thread time), _published_frames accumulates one LiveFrame reference per un-consumed idx. Each retained LiveFrame in live mode still holds its ~16MB (upcast ~64MB float64) map_raw because worker_process only frees raw in batch mode. A sustained 500-1000-frame backlog pins ~30-60GB of raw -> OOM/crash. Bounded only by the GUI keeping pace; there is no cap, drop, or latest-wins on this map.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The mechanism is REAL and unbounded; magnitude depends on GUI-thread backlog (timing), so PLAUSIBLE (per the ladder, do not refute for load-dependence).

PRODUCER (writer QThread), qt_nexus_sink.py `_publish_display`:
  L164-165: `if getattr(self._host, "batch_mode", True): return` — live-only.
  L166-169: `idx = int(live.idx); published = ...; published[idx] = live` — one DISTINCT LiveFrame ref stored per idx (no latest-wins; each frame is a new key).
  L173: `sig.emit(idx)` — one emit per frame, NO producer-side coalescing.
`sig` is the wrangler thread's `sigUpdate`. wranglerThread IS a QThread (wrangler_widget.py:346 `class wranglerThread(Qt.QtCore.QThread)`), so the chain thread.sigUpdate → sigUpdateData.emit (wrangler_widget.py:127) → staticWidget.update_data (static_scan_widget.py:3720 `self.wrangler.sigUpdateData.connect(self.update_data)`) is a cross-thread QUEUED connection: every frame queues exactly one update_data GUI event.

CONSUMER (GUI thread), static_scan_widget.py `update_data`:
  L4090-4093: `frame = getattr(published, "_published_frames", {}).pop(idx, None); if frame is not None: self._pending_frames[idx] = frame` — pops exactly ONE idx per call. So pop rate == emit rate ONLY if the GUI services queued events fast enough.

NO CAP / NO DROP: `_published_frames` is cleared ONLY at run start (image_wrangler_thread.py:597 `self._published_frames.clear()`) and drained ONLY by that single per-idx pop. There is no size cap, no eviction, no latest-wins on this map (verified: the only mutations are :2530 set, :597 clear, and the GUI pop).

RAW IS RETAINED in live mode: qt_nexus_sink.py:205-209 `# PERF-3: free the raw in BATCH mode only ... if getattr(self._host, "batch_mode", True): live.free_raw()` — worker_process frees map_raw ONLY in batch, so each live still holding in `_published_frames` pins its map_raw (frame.py:303 free_raw is the only release path).

MAGNITUDE at 5000 frames: if the writer/reduction pool emits faster than the GUI drains (plausible: heavy render legs + list rebuilds + OS scheduling steal GUI-thread time), the map grows one ref per un-consumed idx. A sustained 500–1000-frame backlog pins 500–1000 × ~16 MB (uint16/uint32 on-disk) — and because the raw is read as float64 upstream (image_wrangler_thread.py:2970 `np.asarray(read_image(...), dtype=float)`), ~64 MB each → ~32–64 GB of raw pinned → OOM/crash. The map is bounded ONLY by the GUI keeping pace; there is no drop/cap/latest-wins.

Caveat keeping this PLAUSIBLE not CONFIRMED: whether the GUI actually falls behind for a sustained window depends on machine/render load and Qt event-loop scheduling. The map is self-limiting during any interval where the GUI keeps pace (each queued update_data pops one idx). But nothing in the code prevents an unbounded backlog, and the retained refs are 64 MB raws in live mode — so the failure is reachable, not guarded.  ||  [trigger:PLAUSIBLE] Mechanism confirmed and reachable in live (non-batch) mode; magnitude/exact backlog is timing-dependent, so PLAUSIBLE per the ladder.

UNBOUNDED MAP, NO LATEST-WINS (qt_nexus_sink.py:164-173): `if getattr(self._host,"batch_mode",True): return` (batch is a no-op, so this is LIVE-only); then `published[idx] = live` and `sig.emit(idx)`. One dict entry + one queued sigUpdate per frame; there is no cap, drop, or latest-wins on `_published_frames`. Same pattern at the second live publish site image_wrangler_thread.py:2530-2531 (`self._published_frames[img_number]=frame; self.sigUpdate.emit(img_number)`).

CONSUMER IS QUEUED CROSS-THREAD, IDX-KEYED POP (wrangler_widget.py:127 `self.thread.sigUpdate.connect(self.sigUpdateData.emit)` → static_scan_widget.py:3720 `self.wrangler.sigUpdateData.connect(self.update_data)`). update_data drains via static_scan_widget.py:4091 `frame = getattr(published,"_published_frames",{}).pop(idx, None)` then 4093 `self._pending_frames[idx]=frame`. The pop is keyed on the exact idx of each queued emit, so `_published_frames` only stays drained while the GUI event loop services the queued update_data calls as fast as the writer thread emits them. Retention does not vanish on pop — it moves into `_pending_frames`, reset to `{}` only every ~5/s drain (static_scan_widget.py:4157), so it accumulates a drain-interval of LiveFrame refs between drains and grows further if the queued update_data calls themselves lag behind (the GUI thread is the same thread that runs the heavy `_drain_pending_frames`/waterfall render legs).

RAW NOT FREED ⇒ EVICTED-BUT-REFERENCED FRAMES PIN RAW: live-mode worker_process skips free_raw — qt_nexus_sink.py:205-209 `# free the raw in BATCH mode only ... if getattr(self._host,"batch_mode",True): live.free_raw()`. And LiveFrameSeries eviction only drops the dict ref, never free_raw — frame_series.py:506 `self._in_memory.pop(idx, None)` (stash eviction) / :565 in evict_persisted_beyond_cap. So a frame evicted from the cap-64 `_in_memory` window while still referenced by `_published_frames`/`_pending_frames` keeps its ~16 MB map_raw alive. Any backlog of un-drained refs therefore pins raws with no bound.

COST AT 5000 FRAMES: each retained live LiveFrame holds map_raw read as float64 (np.asarray(read_image(...),dtype=float) at image_wrangler_thread.py:2970/3017/3042 upcasts a 16 MB uint16 image to ~64 MB). A sustained backlog of a few hundred un-drained frames = many GB; a 500-1000-frame backlog ≈ 30-60 GB → OOM/crash. The cap/latest-wins-absence + raw-not-freed facts are directly quotable; the exact backlog depth (hence whether it reaches OOM vs. just multi-GB steady-state) depends on how far the GUI thread falls behind under heavy render legs / OS scheduling — a realistic reachable state in live 2D, so not REFUTED.
```

</details>

---

### [2] 🟠 HIGH · leak-retention · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:387`

**Summary:** _record_store_mode_groups excludes a dropped-2D frame's ('2d', mode) key from the persisted set, but the stored FrameRecord still carries a heavy 2D view (intensity_2d), so that frame's heavy 2D key never becomes persisted and the record is never evictable.

**Failure @ 5000 frames:** A per-frame publication-gate drop (e.g. an all-dummy GI 2D row) puts the frame idx in dropped_2d, so line 387 skips its ('2d', mode) key. But that frame was upserted into the record store with a populated intensity_2d cake. Because _heavy_mode_keys still includes ('2d', mode) while _persisted_modes never gets it, _label_heavy_payload_persisted_locked returns False forever, so _find_evictable_heavy_label_locked never selects it and _thin_record is never called. Each dropped-2D frame permanently pins its full 2D cake. Over a 5000-frame GI scan with a class of frames dropping their 2D row, hundreds of unevictable cakes accumulate -> multi-GB retention / OOM. Bounded only by how many frames drop 2D, with no hard ceiling.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] The finding is real. Chain, all quoted:

1) A publication-dropped 2D row is PERMANENTLY dropped (not re-fed): nexus_writer.py:1093-1097 appends idx to dropped_indices and records it on the cursor: `_cur.dropped.setdefault(prepared["group_path"], set()).add(int(idx))`; nexus_writer.py:1116 `filtered["dropped_indices"] = dropped_indices`; the row is removed via `_drop_integrated_rows` (1127). `_save_to_nexus` surfaces these as `dropped_by_group` (nexus_writer.py:679-687, e.g. "an all-dummy GI 2D row").

2) qt_nexus_sink.py:387 `if idx not in dropped_2d: modes.extend(("2d", mode) for mode in record.results_2d)` — so for a dropped-2D frame the `('2d', mode)` key is NEVER passed to `mark_persisted` (qt_nexus_sink.py:399 `self._record_store.mark_persisted(mode_labels, modes=modes)`).

3) The record in the store STILL carries the heavy 2D cake: scan_session.py:433-437 upserts `FrameRecord.from_view(view, mode_1d, mode_2d)` built from `event.result_2d` at frame-completion time; frame_view.py:218 `intensity_2d = np.asarray(raw_i2d).T` retains the real (transposed) array. This upsert happens at reduction time, BEFORE the writer's per-frame publication gate drops the 2D row.

4) Eviction is gated on the dropped key: frame_record_store.py:61-69 `_heavy_mode_keys` still includes `("2d", mode)` because that view has a heavy payload; frame_record_store.py:363-365 `_label_heavy_payload_persisted_locked` returns True only if `heavy_keys.issubset(self._persisted_modes.get(label, set()))` — the 2D key is never in `_persisted_modes`, so it returns False forever. frame_record_store.py:340-347 `_find_evictable_heavy_label_locked` therefore never returns that label (require_persisted_for_eviction defaults True: image_wrangler_thread.py:2269-2272 builds the store with only max_items=512/max_heavy_items=64, no override), so `_thin_record` (376) never runs on it.

5) The `max_items=512` cap can't reclaim it either: frame_record_store.py:380-385 requires `_label_persisted_locked(candidate)`, which at 353-354 needs ALL mode keys (incl. the never-persisted 2D key) subset of persisted — also False forever.

REFUTATION CHECKS FAIL: no cap/del/guard drops these records. NOT head-blocked (the loop at 341 skips to the next evictable good frame), so good frames still thin — but the dropped-2D frames themselves accumulate with no ceiling.

MAGNITUDE at 5000 frames: each stuck frame pins its full 2D cake (n_chi×n_q float64: ~4 MB for 500×1000, ~8 MB for 1000×1000). In a GI scan where a class of frames sit below the critical angle and produce all-dummy 2D cakes, k stuck frames = k×(4–8 MB). A few hundred → ~1–4 GB; if thousands of a 5000-frame scan drop 2D → tens of GB → OOM. Bounded only by how many frames drop 2D, with no hard ceiling. Regression risk is proportional to GI-below-horizon frame count.  ||  [trigger:CONFIRMED] Full chain is constructible from source.

TRIGGER (per-frame 2D publication drop, live+batch streaming path):
- nexus_writer.py:1086-1097 `_filter_prepared_output`: a frame whose 2D publication has errors (comment 1031 "all-dummy GI cakes") is appended to `dropped_indices` and, on the append/live path (`not is_replace`), recorded: `_cur.dropped.setdefault(prepared["group_path"], set()).add(int(idx))`. This surfaces back through `_save_to_nexus`'s return.
- qt_nexus_sink.py:420-421: `dropped = self._scan._save_to_nexus(mode=mode)` then `self._mark_record_store_persisted(published, dropped=dropped)` — runs on EVERY flush in the streaming write path (live and batch), so it is on the per-frame hot route, not a one-off.

THE EXCLUSION (qt_nexus_sink.py:385-388):
```
if idx not in dropped_1d:
    modes.extend(("1d", mode) for mode in record.results_1d)
if idx not in dropped_2d:
    modes.extend(("2d", mode) for mode in record.results_2d)
```
The dropped-2D idx's `('2d', mode)` key is omitted from the `modes` list passed to `record_store.mark_persisted` (line 399).

THE RECORD IS STILL HEAVY:
- scan_session.py:410,433-437 upserts every completed frame with `FrameRecord.from_view(view, mode_1d, mode_2d)`; the drop decision happens LATER at write time, so the record was already stored with a populated 2D array.
- frame_view.py:210-218: when `result_2d is not None`, `intensity_2d = np.asarray(raw_i2d).T` (full-size array; an "all-dummy" cake is a present, full-size sentinel-filled array, not None) → frame_record_store.py:27-35 `_view_has_heavy_payload` True → `_heavy_mode_keys` (61-69) includes `('2d', mode)`.

BOTH BOUNDS FAIL TO RECLAIM IT:
- Heavy bound (frame_record_store.py:358-365): `_label_heavy_payload_persisted_locked` requires `heavy_keys.issubset(_persisted_modes[label])`. `('2d',mode)` is heavy but never in `_persisted_modes` → returns False forever → `_find_evictable_heavy_label_locked` (340-347) skips it → `_thin_record` never called on it (376). Its full 2D cake is pinned.
- Items bound (378-394): eviction candidate requires `_label_persisted_locked` (349-356), which needs ALL `_record_mode_keys` (incl. the 2D key) in `_persisted_modes` → also False → the whole record is never popped either.
Store built at image_wrangler_thread.py:2269-2272 with `max_heavy_items=frame_cap`(64), `max_items=512`, default `require_persisted_for_eviction=True` — the persistence guard is active, so neither loop reclaims dropped-2D records.

MAGNITUDE AT 5000 FRAMES: the pinned entries are full 2D cakes, NOT 16 MB raws — each is n_chi×n_q×8B (e.g. a 1000×1000 GI cake ≈ 8 MB). In a GI scan where a class of frames drop their 2D row (the documented all-dummy case), the count of un-thinnable heavy records grows O(number of dropped-2D frames) with no ceiling: e.g. a few hundred dropped frames ≈ a few GB of retained 2D arrays; a persistent drop pattern over the run trends to OOM. No sink-side compensating `_thin`/drop exists (grep of qt_nexus_sink shows only `mark_persisted`, which is the very call that omits the key). This is a genuine full-2D retention leak on the per-frame streaming path, bounded only by how many frames drop 2D.
```

</details>

---

### [3] 🟠 HIGH · leak-retention · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/viewer_raw_lru.py:65`

**Summary:** The hydrated-raw LRU (cap 8) is defeated in image-viewer mode: the keep-set (h5viewer.py:2681-2685) covers every numeric frame_id, so the eviction while-loop finds every candidate in keep_set and breaks without evicting any full-resolution raw. [same root cause also at: /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/h5viewer.py:2073]

**Failure @ 5000 frames:** Not the live-ingest path — user/viewer-driven. In Image Viewer mode, stepping through a 5000-image acquisition hydrates each viewed detector array as float64 (~64MB, 4x upcast from uint16 at h5viewer.py:2073/2145) and evicts nothing, so every distinct viewed frame's raw pins. Loading even a few hundred frames = tens of GB → OOM. In xye/scan mode the cap of 8 (~512MB) holds; the image-mode keep-set removes the ceiling entirely.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:REFUTED] The candidate claims the raw LRU (cap 8) is "DEFEATED in image-viewer mode by the keep set covering every frame_id" so "stepping through thousands of images evicts nothing and pins every viewed raw... -> OOM." This is refuted by the selection-mode guard.

keep is built ONLY from self.frame_ids (h5viewer.py:2682-2685): `keep = tuple(int(label) for label in getattr(self, "frame_ids", ()) ...)`. So the size of keep == the size of the current selection, NOT "every frame ever viewed."

In IMAGE viewer mode the list is SingleSelection (static_scan_widget.py:6002-6006): `if viewer_mode == 'xye': scans.setSelectionMode(QAbstractItemView.ExtendedSelection) ... else: scans.setSelectionMode(QAbstractItemView.SingleSelection)`. Image mode falls in the else -> the user can select at most ONE frame, so frame_ids holds exactly one label and keep is a 1-element set. _data_changed_now confirms frame_ids is the *current* selection, not an accumulation: it clears frame_ids (line 2314), refills from the selected items, then sets `self.frame_ids[:] = loaded_ids` (line 2358).

With keep = {current frame}, the eviction while-loop is fully functional: `while len(order) > limit: stale = next((c for c in order if c not in keep_set), None); ... order.remove(stale); payload["map_raw"] = None` (viewer_raw_lru.py:65-73). It finds prior frames (not the one displayed) and drops their map_raw, holding the cap. VIEWER_RAW_LIMIT = 8 (viewer_raw_lru.py:30). h5viewer.py:2689-2695 additionally pops evicted rows out of viewer_rows_2d/1d in image mode.

Consequence at 5000 frames: stepping through a 5000-image acquisition in Image Viewer holds AT MOST 8 raws. At ~64 MB float64 each (the real upcast at h5viewer.py:2073/2126, np.asarray(..., dtype=float)) that is ~512 MB steady-state, bounded — NOT the "tens of GB -> OOM" the candidate asserts.

The keep-set could only over-protect if frame_ids held many labels, which requires ExtendedSelection — and that is exclusively the `xye` viewer mode. But xye rows carry XYE 1D traces, not 16 MB detector map_raw images, so no large-array OOM arises there either. The candidate's named scenario ("stepping through thousands... evicts nothing") is therefore not constructible: image mode is single-select, so eviction always fires.  ||  [trigger:PLAUSIBLE] The LRU-defeat mechanism is real and quoted, but the candidate's stated TRIGGER ("stepping through" images) is inaccurate; the actual reachable trigger is a multi-select. Cap that DOES exist and normally works: viewer_raw_lru.py:30 `VIEWER_RAW_LIMIT = 8`, enforced by the eviction loop at :65 `while len(order) > limit:` which drops full-res pixels at :73-75 `payload["map_raw"] = None ... payload["bg_raw"] = None`. Defeat: :66-69 `stale = next((candidate ... if candidate not in keep_set), None); if stale is None: break` — when every entry in `order` is in `keep_set`, nothing evicts.

keep-set in image mode (h5viewer.py:2681-2685): `if getattr(self, "viewer_mode", None) == "image": keep = tuple(int(label) for label in getattr(self, "frame_ids", ()) if str(label).lstrip("-").isdigit())`, passed as `keep=keep` at :2687.

4x float64 upcast CONFIRMED at the two viewer insert sites: h5viewer.py:2126 `img_data = np.asarray(read_image(fpath, frame=frame_idx), dtype=float)` and :2073 `img_data = np.asarray(res.image, dtype=float)`, stored at :2145/:2077 `'map_raw': img_data`. A 16 MB uint16 frame becomes ~64 MB float64.

WHY THE CANDIDATE'S "stepping" TRIGGER IS WRONG: `frame_ids` is NOT the full acquisition; it is the currently-SELECTED set, rebuilt on every selection change. h5viewer.py:2314 `_data_changed_now`: `if not show_all: self.frame_ids.clear()` then :2328 `self.frame_ids += sorted([str(item.text()) for item in items])` where `items = self.ui.listData.selectedItems()` (:2315). Single-frame stepping → frame_ids={one frame} → keep_set={one} → the LRU correctly trims to 8. So stepping does NOT pin every raw; the candidate's magnitude ("Loading even a few hundred frames = tens of GB") does not follow from stepping.

WHY IT IS STILL PLAUSIBLE (reachable via multi-select): listData is ExtendedSelection at all times (h5viewer.py:664 `self.ui.listData.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)`), and the image viewer populates one row per frame (:1996 `labels = list(range(1, nframes + 1))`, ~5000 rows). A select-all / shift-range over those rows makes `frame_ids` the whole selection, and the on-demand load loop hydrates every one with no cap (:2347-2358 `for idx_str in idxs: if idx not in self.viewer_rows_2d: self._load_single_frame(...)`), each call invoking `_remember_viewer_raw_lru` while `keep` = the entire selection → break-without-evict → OOM.

MAGNITUDE at scale: selecting the full 5000-frame stack in the Image Viewer hydrates 5000 × ~64 MB float64 ≈ 320 GB with zero eviction → immediate OOM; even a ~500-frame range select ≈ 32 GB → OOM. This is viewer/user-driven, NOT the 5000-frame live/batch waterfall ingest path (the candidate concedes this), so it is off the hot per-frame route — hence PLAUSIBLE, not CONFIRMED: reachability depends on a user multi-selecting many frames, and the candidate's own single-step framing does not construct the failure.
```

</details>

---

### [4] 🟠 HIGH · quadratic · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xrd_tools/session/display_logic.py:637`

**Summary:** accumulate_waterfall vstacks the new row onto the ENTIRE prior 1D-trace stack every flush (np.vstack([base_rows, add])), re-copying all accumulated rows each tick. [same root cause also at: src/xrd_tools/session/display_logic.py:637, src/xrd_tools/session/display_logic.py:637]

**Failure @ 5000 frames:** O(n^2) trace-byte churn over the run: at frame 5000 each late flush re-copies a ~5000x(1-5k) float64 array (~40-200 MB), compounding into a per-tick GUI-thread slowdown / alloc drag near end of scan. Working set stays O(n) (~20-100 MB), so not OOM — a late-scan stall/jank, not a crash. Fix: append into a preallocated/geometric-growth buffer (or a list of rows stacked once at render) instead of re-vstacking the whole history per frame.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] display_logic.py:635-637 in accumulate_waterfall: `if add: add = np.atleast_2d(np.asarray(add, dtype=float)); new_rows = (np.vstack([base_rows, add]) if base_rows.size else add)`. `base_rows = history.rows` (line 610) is the ENTIRE prior accumulated stack — the docstring at lines 421-424 confirms there is NO cap ("The accumulator instead retains every row it has captured"), and count/rows grow O(n_frames) (WaterfallHistory.rows is (n, len(x)), line 444). np.vstack([base_rows, add]) allocates a fresh (n_prev+n_add)×npt array and copies every prior row each call — no preallocation, no in-place/geometric-growth buffer. Append-only within one reset_key, and on the hot live auto-last path each flush adds a new frame (grid-identity reset_key stays stable, lines 598/606), so the vstack fires per flush. NOT REFUTED: waterfall_display_rows (line 647) caps only the DRAWN rows (MAX_WATERFALL_PAYLOAD_ROWS/256), it does NOT bound this accumulator's rows. Magnitude at 5000 frames: npt≈1000-5000 float64 → row=8-40KB; base_rows≈5000×npt → each late flush re-copies ≈40-200 MB. Summed over the run Σk·rowbytes (k=1..5000) = O(n²) ≈ 100-500 GB of TRANSIENT allocation/copy over the run (freed each tick). Working-set stays O(n): resident accumulator ≈20-100 MB (trace-sized, per the brief this is acceptable — NOT OOM). Consequence = a compounding per-tick GUI-thread CPU/alloc drag near end of scan (late-scan jank/stall), not a crash. The candidate's cap `add`-empty guard at lines 638-639 only skips ticks with no new id; the growing-selection live tick always appends. Fix as candidate states: append into a preallocated/geometric buffer (or list-of-rows stacked once at render) instead of re-vstacking full history each flush.  ||  [trigger:CONFIRMED] display_logic.py:637 `new_rows = (np.vstack([base_rows, add]) if base_rows.size else add)` — `accumulate_waterfall` re-copies the ENTIRE prior stack (`base_rows`) every call, while `add` (lines 618-633) holds only the not-yet-captured frames (`if key in have: continue`). So each tick's cost is O(rows already accumulated), not O(new rows).

REACHABILITY (per-frame live/batch waterfall route, CONFIRMED): the accumulator is grown across renders, not per-call fresh. display_publication.py:719 `return self._overlay_waterfall_payload(state)` is the Overlay/Waterfall payload branch; _overlay_waterfall_payload calls `accumulate_waterfall(prior, ...)` (display_publication.py:954-957) passing `prior = widget._waterfall_history` (line 887); the returned history is stored back at display_frame_widget.py:1922-1923 `if history is not None: self._waterfall_history = history` so the NEXT render appends onto the full prior stack. reset_key is grid-identity, NOT generation (display_logic.py:426-434 docstring; line 598 `history.reset_key != reset_key`), so live auto-last selection growth APPENDS every tick rather than resetting — the vstack keeps growing across the whole 5000-frame run. This is the default 1D Overlay/Waterfall live+batch display mode, not viewer-only, not gated off.

MAGNITUDE at 5000 frames: rows are float64 traces on the shared grid (`np.asarray(rows, dtype=float)`, line 636). At npt≈1000-5000, a full-history row block is 5000×1000×8B ≈ 40 MB up to 5000×5000×8B ≈ 200 MB. Each late-scan render re-allocates and copies that whole block for a single added row. Summed over the run this is O(n²) trace-byte churn (~hundreds of MB to low-GB of cumulative transient allocation). CONSEQUENCE: working set stays O(n) (the WaterfallHistory holds one stack ~20-100 MB), so NOT OOM — this is a compounding per-tick GUI-thread CPU/alloc drag that worsens toward end of scan (late-scan jank/stall), exactly a super-linear per-frame slowdown, NOT a raw-image/full-2D retention leak. The 0.5s waterfall-IMAGE throttle (display_plot.py) does not gate this: accumulate_waterfall runs on every payload build/render, not only on image repaints. NO CAP mitigates the vstack cost — MAX_WATERFALL_PAYLOAD_ROWS=256 / waterfall_display_rows (line 969) decimate only the DRAWN rows; the accumulator itself is deliberately uncapped ("retains every row it has captured", display_logic.py:424). Fix as stated (geometric-growth buffer / stack-once-at-render) is valid.
```

</details>

---

### [5] 🟠 HIGH · copy-bloat · PLAUSIBLE

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:2970`

**Summary:** get_next_image reads each raw as np.asarray(read_image(...), dtype=float) — upcasts a uint16/uint32 16 MB detector frame to float64 (4x).

**Failure @ 5000 frames:** Every raw image becomes ~64 MB float64 the instant it enters the pipeline (single-image and directory-series paths, lines 2970 and 3022). frame_from_live_frame aliases this same array as Frame.image (reduction.py:694), and in LIVE mode it is never freed (rides on the LiveFrame). Combined with the parallel worker pool + prefetch queue(4) + 64-frame in-memory stash all holding float64 copies, steady-state raw residency is ~4x the on-disk byte count: a uint16 80 GB scan has a ~4 GB live-mode raw working set instead of ~1 GB. Transient per-frame, but the 4x multiplier applies to every in-flight/retained raw. Fix: keep the native dtype for the raw (defer float conversion into the integrator, which pyFAI does internally) or mask/correct in the source dtype.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The upcast is REAL on the cited lines but does NOT cover "every raw" as claimed — reachability/magnitude depends on the detector/source path.

CONFIRMED-real for the non-Eiger paths: image_wrangler_thread.py:2970 (single-image) `img_data = np.asarray(read_image(self.img_file), dtype=float)` and :3022 (directory-series) `data = np.asarray(read_image(fname), dtype=float)`. `dtype=float` is float64 (8B). A native uint16 16 MB frame → 64 MB (4x); uint32 16 MB → 32 MB (2x). For a 5000-frame per-file directory series (e.g. Pilatus CBF/TIFF), every frame hits :3022 and is upcast the instant it is read. Series-average even accumulates in float64 (`img_data += data`, :3037).

REFUTES the "every raw" universality for the primary streaming source: the Eiger HDF5 path — the canonical high-frame-count streamer, and the one the code itself names for this scenario — does NOT upcast. Lines 2940 `img_data = np.asarray(_raw)`, 2943 `img_data = np.asarray(self._eiger_h5_dataset[frame_idx])`, 2948 `img_data = np.asarray(_raw)` all omit `dtype=float`, preserving native dtype. reduction.py:696 explicitly sizes the retained raw as "~18 MB/frame Eiger" (i.e. ~native uint32/uint16, NOT ~64 MB float64) — direct evidence the streamed Eiger raw is not the 4x-bloated array the candidate describes.

Magnitude at 5000 frames: TRANSIENT per-frame, not accumulated (the finding concedes this). The 4x multiplier applies only to in-flight/retained raws on the non-Eiger path: with the bounded caps (prefetch queue 4, parallel pool width, LiveFrameSeries._in_memory cap 64 at frame_series.py:454), live-mode steady-state raw residency on a uint16 directory series is ~64×64 MB ≈ 4 GB vs ~1 GB native — a real ~3 GB extra working set, bounded (not OOM by itself, not unbounded growth). On the Eiger path the extra is ~0. So: real copy-bloat on lines 2970/3022 for per-file series detectors, but NOT on the Eiger streaming path, and never unbounded. Correctness caveat: the `dtype=float` on non-Eiger paths is largely redundant since pyFAI casts internally during integration, so deferring the cast into the integrator would remove it.  ||  [trigger:REFUTED] The finding claims the xdart lines `np.asarray(read_image(...), dtype=float)` at image_wrangler_thread.py:2970 (single-image path) and :3022 (directory-series path) "upcast a uint16/uint32 16MB detector frame to float64 (4x)." This is factually wrong about the cited lines: `read_image` ALREADY returns float64. In src/xrd_tools/io/image.py the reader does `arr = arr.astype(float, copy=False)` (line 224) unconditionally, and its docstring states the return is a "Float64 image array, NaN where masked or above threshold" (line 185). Therefore `np.asarray(already_float64_arr, dtype=float)` is a NO-OP — np.asarray with a matching dtype returns the same array and copies nothing. The 4x upcast does NOT happen at the cited xdart lines; it happens inside read_image:224, driven by the NaN-masking contract (`arr[arr > threshold] = np.nan` line 227, `arr[mask] = np.nan` line 229 both require float).

The PATH is genuinely reachable — get_next_image feeds img_data into the per-frame collect loop (image_wrangler_thread.py:1023, 1210) which builds `LiveFrame(img_number, img_data, ...)` (lines 1444/2095/2352) on every batch/live/serial frame, default config, exactly the 5000-frame scenario; and reduction.py:693-694 confirms Frame.image aliases LiveFrame.map_raw. And the SUBSTANTIVE consequence (raw residency in RAM is ~4x the uint16 on-disk byte count; a uint16 80GB scan has a ~4GB live-mode raw working set) is REAL. But the finding is anchored to the wrong lines and mischaracterizes them as the upcast site.

The proposed fix — "keep the native dtype for the raw... defer float conversion into the integrator" at the cited lines — is NOT achievable there: removing `dtype=float` from the np.asarray calls changes nothing because read_image hands back float64 regardless. Any real fix would have to change read_image's mandatory float64+NaN-mask contract in ssrl_xrd_tools (a different file, different subsystem, with correctness implications for NaN masking), not the xdart lines this finding cites. Verdict is REFUTED because the specific claim ("these lines upcast 4x") is constructibly false from image.py:185/224.

Magnitude note for completeness: at 5000 frames the float64 residency is bounded, not accumulated — transient per in-flight frame (pool width + prefetch 4) plus the cap-64 _in_memory stash, i.e. ~64 x ~64MB ≈ 4GB steady-state in live 2D. That is the real (bounded) cost, but it originates at image.py:224, not image_wrangler_thread.py:2970/3022.
```

</details>

---

### [6] 🟡 MEDIUM · copy-bloat · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_plot.py:1175`

**Summary:** update_wf copies the ENTIRE growing accumulator (data_.copy()) before slicing/decimating to <=256 rows.

**Failure @ 5000 frames:** Every waterfall repaint deep-copies the full plot_data[1] stack (~5000x(1-5k) float64 ~= 40-200 MB) even though only <=256 rows are ultimately displayed. Throttled to ~2/sec while processing, so ~2 x 40-200 MB transient copies/sec at end of scan — a steady GC/alloc drag on the GUI thread, not OOM. Fix: slice self.plot_data[1][wf_start:wf_stop:wf_step] BEFORE copying (or pass a view into waterfall_display_rows), so only the <=256 displayed rows are ever materialized.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] display_plot.py:1174-1183 proves the copy-before-slice: line 1174 `xdata_, data_ = self.plot_data`; line 1175 `s_xdata, data = xdata_.copy(), data_.copy()` copies the FULL accumulator; only AFTER, line 1180 `data = data[self.wf_start:self.wf_stop:self.wf_step, :]` slices and line 1182-1183 `waterfall_display_rows(data, row_ids, MAX_WF_ROWS)` decimates to MAX_WF_ROWS=256 (display_plot.py:46). So all ~5000 rows are materialized by .copy() before anything is thinned.

The copied object is the unbounded run accumulator: display_plot.py:154 `plot_data[1] = np.vstack((old_y, row))` appends one 1D trace row per frame (no cap on this legacy triple), so at frame 5000 `plot_data[1]` is ~5000×npt float64.

MEMORY/TIME AT 5000 FRAMES: a 1D azimuthal trace of ~1000-5000 pts × 5000 rows × 8B (float64) = ~40-200 MB copied per repaint, while the displayed slice is only ≤256 rows (~2-10 MB). update_wf is throttled to ~0.5s (display_plot.py:1165-1170 `_processing_active` gate) => ~2 repaints/sec => ~80-400 MB/sec of transient GUI-thread allocation at end of scan. NOT OOM (single transient copy, freed after decimation), but a real steady alloc/GC drag that grows linearly with frames. No cap/guard prevents the full copy — the ≤256 bound is applied only downstream of it. (Same copy-before-slice pattern also at update_wf_pmesh display_plot.py:1206, not cited.) Fix (slice `data_[wf_start:wf_stop:wf_step]` before .copy()) is valid.  ||  [trigger:CONFIRMED] display_plot.py:1174-1175 copies the ENTIRE accumulator before any slice/decimate:
  `xdata_, data_ = self.plot_data`
  `s_xdata, data = xdata_.copy(), data_.copy()`
Only AFTER the full copy is the data sliced (line 1180 `data = data[self.wf_start:self.wf_stop:self.wf_step, :]`) and decimated to MAX_WF_ROWS=256 (line 1182-1183 `data, row_ids, _stride = waterfall_display_rows(data, row_ids, MAX_WF_ROWS)`; MAX_WF_ROWS=256 at :46). So `data_.copy()` materializes the whole growing stack even though <=256 rows are ever displayed.

REACHABLE on the hot path: update_wf is called from update_plot_view at :932 (`if auto_wf: self.update_wf()`), whose auto_wf trigger fires precisely for a many-curve Waterfall/Overlay selection (:927-928 `plotMethod == 'Waterfall' and n_curves > 3` / `not in Sum/Average and n_curves > 15`). plot_data[1] IS the full per-frame accumulator: :915 `n_curves = len(self.plot_data[1])` is used as the drawn frame count and the :919-921 comment states "the payload path emits one un-reduced trace per frame, so n_curves is the frame count". This is the default live/batch waterfall route, not viewer-only.

MAGNITUDE at 5000 frames: plot_data[1] is ~5000 x npt float64. At npt=1000 that is 5000*1000*8 = 40 MB; at npt=5000, 200 MB. update_wf copies this whole array every repaint. Throttled to >=0.5s while _processing_active (:1165-1170 `if now - _wf_last_draw_t < 0.5: return`), i.e. ~2 repaints/sec, so ~2 x 40-200 MB transient copies/sec at end of scan = a steady GC/alloc drag on the GUI thread. NOT OOM (single transient copy, freed after the function) and NOT raw-image/full-2D retention — it is trace-sized copy-bloat (scenario b). Identical pattern in the sibling update_wf_pmesh at :1205-1206 (unthrottled). Fix as stated: slice/stride BEFORE copying (self.plot_data[1][wf_start:wf_stop:wf_step] then pass to waterfall_display_rows) so only <=256 rows are materialized.
```

</details>

---

### [7] 🟡 MEDIUM · quadratic · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/static_scan_widget.py:4220`

**Summary:** _scan_info_rows grows one coerced scan_info dict per frame for the whole run and is only cleared at a scan boundary; each drain then rebuilds the ENTIRE scan_data DataFrame from all accumulated rows (pd.DataFrame.from_dict over all N rows + sort_index) every ~200ms. [same root cause also at: /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/static_scan_widget.py:4220]

**Failure @ 5000 frames:** Two compounding costs over the run: (1) _scan_info_rows retains N dicts (metadata-sized, few MB at 5000 -- not the killer). (2) Every drain rebuilds scan_data from ALL accumulated rows, so the per-drain DataFrame build is O(N) and the aggregate over ~N/interval drains is O(N^2) pandas work on the GUI thread. At 5000 frames the late-scan drains each rebuild a 5000-row heterogeneous DataFrame + sort every 200ms, adding a growing per-drain GUI-thread cost that worsens the backpressure/stall as the scan progresses.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] The candidate is real as written. `_scan_info_rows` is a dict that grows one coerced metadata dict per frame and is cleared ONLY at a scan boundary: constructed at static_scan_widget.py:3634 `self._scan_info_rows = {}`, appended per frame at :4220 `self._scan_info_rows[int(idx)] = coerced`, and cleared only at :3634 (init) and :5494 (scan boundary, `self._scan_info_rows = {}` in the new_scan reset). No mid-run trim exists (grep shows only those four sites).

Every drain rebuilds the ENTIRE scan_data DataFrame from ALL accumulated rows (:4230-4233):
```
df = pd.DataFrame.from_dict(self._scan_info_rows, orient="index")
df.sort_index(inplace=True)
with self.scan.scan_lock:
    self.scan.scan_data = df
```
This runs inside `_drain_pending_frames` (:4142), which the design comment (:3625-3632) states runs "once per ~200 ms flush over ALL frames stashed since the last flush" on a throttled Coalescer (`self._update_timer = Coalescer(_flush_ms, mode="throttle")` :3613 → `_flush_pending_update` → drain). The comment at :4223-4225 explicitly acknowledges the build is "O(N) per flush" (chosen over the prior per-frame `sd.loc[idx]=ser` O(N^2) enlargement) — but because `_scan_info_rows` is never trimmed, that O(N)-per-flush build is over the growing full row set, so aggregate over ~N/interval drains is O(N^2) pandas work on the GUI thread.

Magnitude at 5000 frames — the retention is NOT the killer: `_scan_info_rows` holds ~5000 small heterogeneous dicts = a few MB (metadata-sized, no raw images / no full-2D arrays — confirmed by inspection: `coerced = _coerce_scan_info(info)` at :4218 holds scalar motor/scan fields only). The real cost is CPU: each late-scan drain rebuilds a full ~5000-row heterogeneous DataFrame + `sort_index` every ~200ms. `pd.DataFrame.from_dict(orient="index")` over N heterogeneous dicts is O(N) per call (with pandas per-column dtype inference, in practice worse constant); summed over the ~N/(frames-per-flush) drains → O(N^2) GUI-thread pandas work, a per-drain cost that grows monotonically through the run and compounds the display backpressure/stall late in a 5000-frame scan. The `XDART_PERF` `scan_data=%.0fms` leg (:4237/4241-4244) is instrumented precisely to observe this growth. Verdict CONFIRMED: the quadratic per-drain rebuild is real and reachable on the normal live path (no guard bounds `_scan_info_rows` mid-run); the memory retention is bounded/metadata-sized as the candidate itself notes.  ||  [trigger:CONFIRMED] Reachable on the hot live path: update_data pops each integrated frame into self._pending_frames (static_scan_widget.py:4093); the default 150ms throttle _update_timer (Coalescer(_flush_ms...), :3613) fires _flush_pending_update → self._drain_pending_frames() (:4377). Default config, exactly the 5000-frame live/batch route.

RETENTION (metadata-sized, correctly NOT the killer): line 4220 `self._scan_info_rows[int(idx)] = coerced` accumulates one coerced scan_info dict per frame; the dict is only cleared at a scan boundary (`self._scan_info_rows = {}` at :3634 init and :5494). At 5000 frames = ~5000 dicts, a few MB — trace/metadata-sized, not raw-image/2D, so not OOM.

SUPER-LINEAR GUI-THREAD WORK (the real consequence): every drain with new rows rebuilds the ENTIRE table from all accumulated rows — line 4230 `df = pd.DataFrame.from_dict(self._scan_info_rows, orient="index")` then :4231 `df.sort_index(inplace=True)`, assigned under `with self.scan.scan_lock:` (:4232-4233). This is O(N) heterogeneous-dtype pandas construction + full sort per drain; over ~N/flush drains the aggregate is O(N²) pandas work on the GUI thread. At 5000 frames the late-scan drains each build+sort a 5000-row heterogeneous DataFrame every ~150ms, a growing per-drain cost that compounds the backpressure/stall as the scan progresses.

Cost quantified at 5000 frames: memory ≈ 5000 small dicts (few MB, negligible); time ≈ O(N²) DataFrame rebuilds — the final drains each rebuild a full 5000-row×(n_columns) heterogeneous DataFrame + sort_index on the GUI thread under scan_lock, i.e. a per-drain O(N) tens-of-ms hit at N=5000 that did not exist early in the scan, worsening liveness monotonically.

Mitigation noted (does not refute — softens magnitude): the code comment at :4223-4225 states this is the DELIBERATE replacement for a worse O(N²)-per-frame `sd.loc[idx] = ser` enlargement, so the rebuild is O(N) per DRAIN not per FRAME (constant reduced by frames-per-drain). But it is still whole-table-per-drain, so the run aggregate stays O(N²). Real compounding GUI-thread slowdown, not OOM.
```

</details>

---

### [8] 🟡 MEDIUM · quadratic · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/metadata.py:154`

**Summary:** metadataWidget.update() transposes the FULL scan_data table every visible refresh when the live selection is Overall. `_selected_scan_data(sd)` returns `scan_data.loc[present]` for ALL N frames (or the whole table when frame_ids is empty, lines 109/119), and `.transpose()` then materializes an M-motor x N-frame DataFrame into a fresh DFTableModel on EACH call. update() is driven by the coalesced set_data/sigUpdate path (static_scan_widget.py:4612, ~few/sec during a live scan), and the panel is normally visible, so the transposed table it rebuilds grows one column per frame across the run. This is a per-frame full-rebuild disguised as an incremental 'update selected rows' refresh.

**Failure @ 5000 frames:** At frame 5000 with ~20 motor columns and the metadata panel visible + Overall selected, every refresh transposes a ~5000x20 table (~100k cells) and rebuilds the Qt table model; summed over the run's refreshes this is O(N^2) GUI-thread CPU (a compounding late-scan drag / stutter, seconds of cumulative transpose work), with transient metadata-sized (~sub-MB) copies each tick. Not OOM — a CPU/responsiveness regression that worsens as the scan lengthens.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] Mechanism is real and I can name the exact state that triggers it. metadata.py:154 rebuilds the full transposed table every visible refresh:

  metadata.py:154  self.tableview.setModel(DFTableModel(self._selected_scan_data(sd).transpose()))

`_selected_scan_data` returns the FULL table when frame_ids covers all frames:
  metadata.py:117-120  present = [label for label in labels if label in scan_data.index]; ... return scan_data.loc[present]
  (and metadata.py:109-110 returns the whole `scan_data` outright when frame_ids is empty).

During a LIVE Overlay/Waterfall scan, frame_ids is populated with EVERY processed frame, not a small selection:
  static_scan_widget.py:4505  selected = [str(int(i)) for i in self.scan.frames.index]   # ALL N frames
  static_scan_widget.py:4507  self.h5viewer.frame_ids[:] = selected
The metadataWidget shares the SAME list object (both constructed from self.frame_ids: static_scan_widget.py:535 for H5Viewer and 674-675 for metawidget; line 4507 mutates it in place), so `present` = all N labels → `scan_data.loc[<all N>]` (a copy) → `.transpose()` (a second M×N copy) on every call.

update() is driven per live refresh via the coalesced flush; the code path is explicitly annotated:
  static_scan_widget.py:4418  staticWidget._h5viewer_data_changed_now(self)  # -> sigUpdate -> set_data -> metawidget.update()
  static_scan_widget.py:4612  self.metawidget.update()
Full-scan reselect + this refresh fire at ~2 Hz during a live run (throttle at static_scan_widget.py:4406-4416, `if now - last < 0.5`), and sub-throttle ticks still call metawidget.update() with the same all-N frame_ids.

The only guard is the visibility gate:
  metadata.py:145  if not self.tableview.isVisible(): return
This REFUTES the cost only when the metadata panel is hidden; the panel is normally on screen during a live scan, so the cost is realized. DFTableModel itself is lazy (gui_utils.py:121-124 renders cells on demand via iloc), so the model construction is cheap — the cost is the pandas .loc fancy-index copy + .transpose() materialization, NOT retention.

Magnitude at 5000 frames (~20 motor columns): each visible refresh copies+transposes a ~5000×20 ≈ 100k-cell table (two sub-MB DataFrame allocations), i.e. O(N·M) GUI-thread work per tick. Summed over the run's ~N/throttle refreshes this is O(N²) transpose work — a compounding late-scan stutter (seconds of cumulative transpose time by frame 5000), with only transient sub-MB metadata-sized copies each tick. This is a CPU/responsiveness regression, NOT raw-image/2D retention and NOT OOM — exactly the "quadratic, medium" class the candidate claims. Minor imprecision in the candidate: the live-overlay path fills frame_ids (hitting metadata.py:120), not the empty-frame_ids branch (metadata.py:119); the outcome (full N-column transpose per tick) is identical.  ||  [trigger:PLAUSIBLE] The full-table transpose per visible refresh is real and reachable on the live path.

metadata.py:154: `self.tableview.setModel(DFTableModel(self._selected_scan_data(sd).transpose()))` — gated only by `viewer_mode is None and sd is not None and len(sd.index) and len(sd.columns)` (lines 148-153) and by panel visibility (`if not self.tableview.isVisible(): return`, line 145). No incremental / diff path; every call rebuilds a fresh DFTableModel and calls setModel (full view reset).

`_selected_scan_data` returns the WHOLE N-frame table on the hot path:
- metadata.py:109-110 `if not self.frame_ids: return scan_data` (the empty-selection / Overall branch), AND
- metadata.py:120 `return scan_data.loc[present]` where `present` is EVERY frame currently in scan_data.

The candidate's stated trigger ("Overall → frame_ids empty") is slightly imprecise but the consequence holds via a DIFFERENT populated path: in live Overlay/Waterfall + auto_last, `_render_overlay_full_scan` sets `self.h5viewer.frame_ids[:] = [str(int(i)) for i in self.scan.frames.index]` — ALL N frame ids (static_scan_widget.py:4505-4507) — then `_h5viewer_data_changed_now(show_all=True)` → sigUpdate → `set_data` (connected at static_scan_widget.py:3426) → `self.metawidget.update()` (static_scan_widget.py:4612). So `frame_ids` holds all N and line 120 returns `scan_data.loc[all N]`, transposed. Either branch transposes the full M×N table.

DFTableModel just stores the frame (gui_utils.py:115-119), so the cost is the pandas `.transpose()` materializing an M-motor × N-frame table each call; `data()` (line 124) is lazy per visible cell but the transpose copy is not.

Reachability/rate: this is on the per-flush live route, but throttled to >=0.5s during a live scan (static_scan_widget.py:4406-4413, the 0.5s `_overlay_flush_last_t` gate), so ~2 Hz, not literally per-frame; the metadata panel is default-visible. The transposed table grows one column per frame, so at frame 5000 with ~20 motor columns each refresh transposes a ~5000×20 (~100k-cell) table and rebuilds the Qt model; summed over the run this is O(N^2) GUI-thread transpose/model-rebuild work — a compounding late-scan stutter with sub-MB transient metadata-sized copies. NOT OOM and NOT raw-image/2D retention (scan_data is per-frame floats/strings). Consequence = CPU/responsiveness regression that worsens as the scan lengthens, exactly as claimed.

PLAUSIBLE rather than CONFIRMED: exact magnitude depends on live frame rate, motor-column count, and the panel being visible; the ~2 Hz throttle (not per-frame) softens but does not remove the O(N^2) growth.
```

</details>

---

### [9] 🟡 MEDIUM · quadratic · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/display_publication.py:903`

**Summary:** _overlay_waterfall_payload re-fetches, re-normalizes and re-interpolates EVERY resident publication in state.render_ids each tick before accumulate_waterfall dedups. [same root cause also at: src/xdart/gui/tabs/static_scan/display_controllers.py:336]

**Failure @ 5000 frames:** The `for label in state.render_ids` loop pulls each publication, runs _normalize + _apply_plot_unit_1d + a shape check + np.interp per row every render, even though only the newly-arrived frame is actually appended (dedup lives downstream in accumulate_waterfall). render_ids tracks the store's resident light tier (bounded ~512), so each tick redoes up to ~512 fetch+normalize+interp cycles; across ~5000 ticks that is millions of redundant per-row normalizations on the GUI thread — wasted O(store_cap) work per tick that compounds into a visible late-scan slowdown. Would be strictly O(n^2) if the light tier were unbounded.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The MECHANISM is real but the MAGNITUDE (~512, "millions", "O(n^2)") is overstated; the true cost is a bounded O(~64) redundant per-tick constant, not compounding.

REAL redundancy: display_publication.py:903 `for label in state.render_ids:` re-does per resident frame every render — line 915 `y = self._normalize(y, pub.metadata_raw)`, line 916-917 `x, conv_axis = self._apply_plot_unit_1d(...)`, and the reinterp at line 931 `y = np.interp(ref_x, x, y)` — then accumulate_waterfall (line 954) DISCARDS all but genuinely-new frames via its dedup set: display_logic.py:616 `have = {_dedup_key(i) for i in history.ids}` and 622-623 `if key in have: continue`. So on a live tick where 1 new frame arrived, the ~N-1 already-accumulated frames are normalized+converted+interpolated for nothing. Redundant work confirmed.

The cap that REFUTES the "~512 / O(n^2)" magnitude: state.render_ids = resolve_render_ids(...) ∩ loaded_1d (display_logic.py:1654,1693; resolve_render_ids:866 `i in loaded`). loaded_1d only adds a frame when its ONE_D read is RESIDENT with `view.has_1d` — display_publication.py:909 `if not view.has_1d ... continue`. But the store's eviction STRIPS the 1D array from every frame past the heavy window of 64: frame_publication.py:822 `self._items[label] = _semilight_publication(publication)` fires when `len(self._heavy_labels) > self._max_heavy_items` (max_heavy_items=64, line 535), and _semilight_publication builds a `thumb_view` (line 401-413) that carries NO intensity_1d → view.has_1d=False. The lighter tier (line 831 _lightweight_publication) also strips it. max_thumbnail_items(512)==max_items(512) so tier-2 never precedes the full pop; and for INT_1D `include_legacy = mode not in (INT_1D, INT_2D)` = False (display_controllers.py:315), so no legacy data_1d fallback re-inflates loaded_1d. Net: only the ~64 heavy-window frames satisfy has_1d, so render_ids ≈ 64 in steady state, NOT ~512, and NOT growing with the run.

Cost at 5000 frames: ~64 traces × (normalize + Q↔2θ convert + possible interp) of ~1-5k-point 1D arrays, redone once per COALESCED GUI drain (~5/sec, not per frame). That is a fixed ~O(64) redundant constant per tick — a modest wasted-CPU overhead on the GUI thread, flat across the whole run. It does NOT compound into a late-scan slowdown and is NOT O(n^2) (both refuted by the max_heavy_items=64 strip). No memory retention: the loop copies only 1D traces (~few thousand floats), never raw 16MB images or full-2D arrays. PLAUSIBLE as a real but bounded inefficiency; the finding's severity/scaling claims are refuted by the 64-frame heavy cap.  ||  [trigger:CONFIRMED] The finding is confirmed as a bounded per-tick wasted-work / compounding-slowness issue (NOT OOM, NOT strictly O(n^2)), matching its own "quadratic, medium" framing.

REACHABILITY (per-frame live Overlay/Waterfall route): display_publication.py:717-719 dispatches to _overlay_waterfall_payload exactly when `state.mode in (Mode.INT_1D, Mode.INT_2D) and state.method in ("Overlay", "Waterfall")` — the live/batch waterfall path, not viewer-only, not gated off.

REDUNDANT PER-ROW WORK OVER ALL RESIDENT FRAMES (display_publication.py:903-931): `for label in state.render_ids:` pulls each publication and for every one runs `y = self._normalize(y, pub.metadata_raw)` (:915), `x, conv_axis = self._apply_plot_unit_1d(...)` (:916-917), a shape check (:913), and `y = np.interp(ref_x, x, y)` when the grid mismatches (:930-931). Every resident frame is re-fetched/re-normalized/re-interpolated every render.

DEDUP LIVES DOWNSTREAM, so that work is discarded for all-but-the-new frame — display_logic.py accumulate_waterfall: `have = {_dedup_key(i) for i in history.ids}` (:616) then `for i,n,r,m in zip(...): key=_dedup_key(i); if key in have: continue` (:620-623). Only genuinely-new ids append (`np.vstack([base_rows, add])`, :637). The ~N-1 already-accumulated rows the line-903 loop just recomputed are thrown away.

BOUND ON render_ids (the magnitude): render_ids = primary (display_logic.py:1693) = resolve_render_ids(...) = `all_frame_index if overall else selected` ∩ loaded_keys (:864-866). loaded_1d comes from _data_snapshot iterating candidate labels and adding only ReadStatus.RESIDENT frames (display_controllers.py:154-159). Residency is capped by PublicationStore `_max_items = DEFAULT_PUBLICATION_MAX_ITEMS = 512` (frame_publication.py:534,546,809 `while len(self._items) > self._max_items`). So at frame 5000 the loop is ~512 iterations/tick, not 5000.

MAGNITUDE at 5000 frames: once the resident light tier saturates the 512 cap, each of ~5000 ticks redoes up to ~512 fetch+normalize+_apply_plot_unit_1d(+possible np.interp) cycles on the GUI thread whose results the downstream dedup discards — millions of redundant per-row normalizations, a real steady-state GUI-thread CPU drag late in the scan. It is O(store_cap) wasted work per tick (bounded), and would be strictly O(n^2) only if the light tier were unbounded — exactly as the finding states. No raw-image or full-2D retention on this path (traces only).
```

</details>

---

### [10] 🟡 MEDIUM · quadratic · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/display_plot.py:1128`

**Summary:** Time-based waterfall y-axis computes min(_frame_scan_info(idx)['epoch'] for idx in full_row_ids) where _frame_scan_info itself does an O(n) linear scan of history.ids — O(n^2) per repaint.

**Failure @ 5000 frames:** When wf_yaxis is 'Time (s)' or 'Time (minutes)', _wf_y_axis iterates ALL accumulated ids (full_row_ids, up to 5000) to find the baseline, and each _frame_scan_info (display_plot.py:1055) does a `for row_id, info in zip(history.ids, meta)` linear scan to resolve one id. That is n × n = up to ~25M zip-iterations per waterfall repaint at frame 5000 (plus a second full pass to build the per-row epoch array). Throttled to ~2/sec during live, but each repaint's cost grows quadratically with frames accumulated → the time-axis waterfall repaint gets progressively jankier and can exceed the 0.5 s throttle late in the scan.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] REAL and quadratic on the Time-axis waterfall path. In `_wf_y_axis` (display_plot.py), `full_row_ids` is built from the FULL accumulated history, not the decimated draw set: line 1096-1097 `full_row_ids = tuple(getattr(history, "ids", ()) or ())` — up to 5000 ids at frame 5000. For the time axes the baseline scans ALL of them: line 1127-1128 `baseline = min(self._frame_scan_info(idx)['epoch'] for idx in full_row_ids)` (and the identical line 1135-1136 for 'Time (minutes)'). Each `_frame_scan_info(idx)` (line 1044) resolves ONE id by a linear scan over the same history: line 1055 `for row_id, info in zip(history.ids, meta): if row_id == idx and info: return dict(info)`. So the baseline alone is 5000 outer × ~2500 avg (5000 worst) inner zip-steps ≈ 12.5M–25M iterations PER repaint at frame 5000; the s_ydata build (line 1129-1132) adds 256×5000 ≈ 1.28M more. This is CPU/time, not memory — no OOM, no retention (the y-axis arrays are trace/scalar sized).

TRIGGER (config-gated, reachable): fires ONLY when `self.wf_yaxis == 'Time (s)'` (line 1126) or `'Time (minutes)'` (line 1134). The default 'Frame #' path is O(n) via a dict `pos_by_id = {row_id: pos+1 ...}` (line 1119), NOT quadratic — REFUTES the concern for the default axis. The arbitrary-counter `else` branch (line 1142-1146) builds no full-history baseline, so it is 256×5000 ≈ 1.28M, linear-ish, not the full quadratic. The `min(... for idx in full_row_ids)` baseline is what makes Time specifically O(n²).

NOT REFUTED by the throttle: the 0.5s guard (line 1165-1170 `if now - getattr(self, "_wf_last_draw_t", 0.0) < 0.5: return`) only limits repaint FREQUENCY (~2/sec live); it does not reduce per-repaint cost. As full_row_ids grows toward 5000, each Time-axis repaint's ~25M-iteration cost rises and can EXCEED the 0.5s budget late in the scan → the waterfall repaint monopolizes the GUI thread and janks/stalls. Consequence at 5000 frames: a compounding per-repaint slowdown on the GUI thread (up to tens of millions of Python zip/dict comparisons per repaint), not memory growth — a real end-of-scan stall on the Time-axis waterfall, no effect on memory or frame correctness.  ||  [trigger:CONFIRMED] O(n^2) per-repaint confirmed for the Time-axis waterfall. `_frame_scan_info(idx)` does a linear scan of the full history to resolve ONE id: "for row_id, info in zip(history.ids, meta): if row_id == idx and info: return dict(info)" (display_plot.py:1055-1057). `_wf_y_axis` builds `full_row_ids` from `history.ids` UNCAPPED — "full_row_ids = (tuple(getattr(history, 'ids', ()) or ()) ..." (1096-1102). For the Time modes the baseline iterates ALL of them: "baseline = min(self._frame_scan_info(idx)['epoch'] for idx in full_row_ids)" (display_plot.py:1128 for 'Time (s)', 1136 for 'Time (minutes)'). n calls x O(n) scan each = n^2.

MAGNITUDE at 5000 frames: `history.ids` grows one id per frame — `accumulate_waterfall` does "out_ids.append(i)" (display_logic.py:625), so full_row_ids reaches ~5000 (no 256 cap applies here; the MAX_WF_ROWS=256 decimation at display_plot.py:1182-1183 caps `row_ids`/the drawn image, NOT `full_row_ids` used for the min()). The baseline pass = 5000 x up-to-5000 zip iterations = ~25M iterations per repaint. The per-row epoch array (1129-1132) adds 256 x O(5000) = ~1.28M more. This is transient CPU per repaint, NOT memory retention.

REACHABILITY: 'Time (s)'/'Time (minutes)' are standard user-selectable options in the waterfall-options dropdown — "counters = ['Frame #', 'Time (s)', 'Time (minutes)']" (display_plot.py:1450), set via "self.wf_yaxis = self.wf_yaxis_widget.currentText()" (1472). `update_wf` is on the live waterfall repaint path (called at display_plot.py:932). The 0.5s throttle (1165-1170) only gates FREQUENCY, not per-call cost, so as the ~25M-iteration cost grows late in the scan it eventually exceeds the 0.5s budget -> the Time-axis waterfall repaint gets progressively jankier and stalls the GUI thread. Default 'Frame #' avoids this (dict lookup, 1119), so the trigger is: user selects a Time axis. CONFIRMED (compounding per-frame slowdown), not an OOM/retention finding.
```

</details>

---

### [11] 🟡 MEDIUM · quadratic · PLAUSIBLE

**Location:** `src/xdart/gui/tabs/static_scan/display_plot.py:154`

**Summary:** The legacy accumulator (update_plot_accumulator) also vstacks onto the whole prior stack each frame (line 154) AND, on x-grid mismatch, takes a union1d + per-row reinterp REBUILD branch (156-164) that rebuilds every prior row per tick. This runs in parallel with the payload accumulator during the migration. [same root cause also at: /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_plot.py:154, /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_plot.py:156, src/xdart/gui/tabs/static_scan/display_plot.py:154]

**Failure @ 5000 frames:** On the hot live path the frozen common grid keeps x uniform, so the cheap allclose APPEND path (line 152) is taken — but it is still an O(n) vstack every frame, doubling the O(n^2) trace-byte churn (this + display_logic's accumulator both maintained). If GI per-frame axes are ever not frozen (per-frame grid mismatch), the union1d REBUILD reinterps all prior rows each tick = hard O(n^2) with a large constant → visible end-of-run stall on a 5000-frame scan. Trace-sized, not OOM.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] CONFIRMED that the legacy accumulator vstacks the whole prior stack every APPEND tick and has no row cap. display_plot.py:152-154 (APPEND branch): `elif old_x.shape == new_x.shape and np.allclose(old_x, new_x): / old_y = _as_plot_rows(plot_data[1]) / plot_data[1] = np.vstack((old_y, row))` — this re-copies ALL accumulated rows into a new array every frame. `_as_plot_rows` (line 66-70) confirms the accumulator is a 2D array of 1D traces (each row = one integrated trace of ~npt floats), NOT raw images — so this is the trace-sized class, not OOM.

STILL LIVE (parallel accumulator, no cap that refutes): update_plot_accumulator is invoked on the hot overlay path at display_plot.py:707-720 (`elif overlay_action is OverlayAction.APPEND:` → update_plot_accumulator(self.plot_data, ...)) and writes back to self.plot_data/self.frame_names/self.overlaid_idxs. display_logic.py:562 documents WaterfallHistory as the "payload-owned successor to update_plot_accumulator + the widget triple" — i.e. both accumulators coexist during migration, as the candidate claims. There is NO cap on plot_data row count in this function (the 256-row MAX_WATERFALL_PAYLOAD_ROWS cap the map cites lives only in the DRAW path / payload, not in this accumulator), so the vstack grows with n_frames — the guard that would refute this does not exist on the accumulator itself.

REBUILD branch is real but guarded: lines 156-164 (`else: merged_x = np.union1d(old_x, new_x) ... new_old = np.array([_reinterp_plot_row(old_x, r, merged_x) for r in old_y]) ... plot_data = [merged_x, np.vstack((new_old, new_row))]`) reinterps EVERY prior row per tick = hard O(n^2). It is only reached when `old_x.shape != new_x.shape or not np.allclose(old_x, new_x)` (the else of line 152), i.e. per-frame x-grid mismatch. On the hot live path the frozen common grid keeps x uniform, so the cheap allclose APPEND branch (154) is taken and REBUILD is not hit — so REBUILD is load/config-gated (fires if GI per-frame axes are ever not frozen), consistent with the candidate.

MAGNITUDE at 5000 frames (npt~1000, float64): each late-run APPEND vstack copies the full ~5000×1000×8 = ~40 MB accumulator once per tick; summed over the run this is O(n^2) trace-byte churn (cumulative ~100 GB of transient allocation traffic) but the WORKING SET stays O(n) ≈ 40 MB (bounded, not OOM). Consequence = a compounding per-tick CPU/alloc drag that worsens toward end-of-scan (a visible GUI stall on the accumulate step), doubled because the payload WaterfallHistory accumulator runs the same vstack in parallel (display_logic.py:635-637). Not refuted: no cap bounds the accumulator rows here; the mechanism is real and reachable. PLAUSIBLE (not CONFIRMED) because the exact per-run slowdown depends on npt and on how many of the 5000 frames actually trigger a distinct accumulate call vs the ~5/s coalesced GUI drain — a timing/config dependence, which the ladder says to rate PLAUSIBLE rather than refute.  ||  [trigger:REFUTED] The cited vstack/union1d lives in `update_plot_accumulator` (display_plot.py:83), whose ONLY caller is the method `update_plot` (display_plot.py:484, called at :681/:695/:709/:724). `update_plot` has ZERO production callsites: `grep -rn "update_plot(" src/xdart/` returns only the docstring at display_plot.py:752; the only real callers are tests — tests/xdart/test_live_refresh.py (15 calls) and tests/xdart/test_aggregation_wiring.py (3 calls). Its own docstring (display_plot.py:487-496) states it is "RETAINED as render_display's PLOT_1D None-payload fallback (a safety net) and as the direct target of the legacy characterization tests," with production control now driven "through `update()` -> the payload pipeline."

The 5000-frame live/batch waterfall route is `update()` -> `render_display` (display_frame_widget.py:2597-2614): for each role it tries the payload (`_draw_payload`, :2599) and `continue`s on success, otherwise calls `_draw_delegate(role, mode)`. For PLOT_1D, `_draw_delegate` returns `self.clear_plot_view` (display_frame_widget.py:1810-1813) — NOT `update_plot`. So neither the payload-success path nor the None-payload fallback reaches `update_plot_accumulator`. The production per-frame accumulator is `_draw_payload` (display_frame_widget.py:1903 `self.plot_data = [ref_x, ydata]`, :1923 `self._waterfall_history = history`), which draws the payload rows already decimated to MAX_WATERFALL_PAYLOAD_ROWS=256 and stores the immutable WaterfallHistory — it never calls the vstack at display_plot.py:154 nor the union1d REBUILD at :155-164.

Therefore the path is unreachable in the 5000-frame live/batch scenario (dead-but-test-exercised legacy code), so the claimed consequences do not follow: the "O(n) vstack every frame doubling the O(n^2) churn" never executes per-tick in production, and the "union1d REBUILD reinterps all prior rows each tick => hard O(n^2) end-of-run stall" is gated behind a method with no production caller. Cost at 5000 frames on the actual production path: 0 bytes/0 ops from update_plot_accumulator; the real accumulator is the payload WaterfallHistory (1D-trace-sized, ~20-100 MB, capped-256-row draw) which the brief classifies as non-alarming. The concern is real only for the legacy tests, not the live/batch scan.
```

</details>

---

### [12] 🟡 MEDIUM · correctness · CONFIRMED

**Location:** `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:3019`

**Summary:** Series-average + Append mode: _should_skip_before_read collapses every source frame to output number 1 (via _append_output_number), so if the target .nxs already contains frame 1 the read-side loop skips ALL source frames (continue) and the running-mean never accumulates.

**Failure @ 5000 frames:** A series-average scan re-run in Append mode (or averaging into a .nxs that already holds a frame 1) silently produces NO output: get_next_image returns (None, None, 1, None, {}) with n never incrementing. All N source frames are dropped-before-read; the averaged frame the user expected is never written. Correctness/dropped-frame class, not scale — but for a 5000-source-frame series average the entire run yields zero data with only an info-level skip log. Uncovered by tests (test_append_skip_before_read.py forces series_average=False).

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The collapse is REAL in code. `_append_output_number` (image_wrangler_thread.py:760-763) hard-returns 1 for series_average regardless of img_number: "if getattr(self, 'series_average', False): return 1". `_should_skip_before_read` (940-947) keys the skip test on that collapsed number: "output_img_number = self._append_output_number(img_number)" then "if output_img_number in self._append_skip_snapshot(scan_name): ... return True". The snapshot (921-938) is the set of output frame-ids already present in the Append target .nxs, primed from disk at run start when `_append_skip_enabled()` (756-758: write_mode=='Append' and not xye_only).

In the series-average read loop (3006-3054): every source frame calls `if self._should_skip_before_read(sname, snumber): continue` (3019-3020). Because ALL source frames collapse to output number 1, if the target already contains frame 1 the snapshot lookup returns True for every one → `continue` fires each iteration → `n` (init 0 at 3007) never increments → the running-mean accumulator branch (3033-3044) is never reached → the loop drains img_fnames and returns `(img_file=None, scan_name=None, img_number=1, img_data=None, {})` at 3054 (init values from 3006). Zero output, only an info-level skip log (958-964). This is a dropped-frame/zero-output correctness bug, NOT a retention/scale bug — memory/time cost at 5000 frames is negligible (no arrays retained; 5000 cheap set-membership skips), but the CONSEQUENCE at 5000 source frames is the entire averaged frame the user expected is silently never written.

REACHABILITY: requires write_mode=='Append' + series_average==True + target .nxs already holding output frame 1 (e.g. re-running a series-average acquisition into an existing Append target). All three are realistic reachable states; exact trigger depends on whether the target already has frame 1, hence PLAUSIBLE rather than CONFIRMED. NOT REFUTED — no guard prevents it: the skip returns True unconditionally on the snapshot hit and the loop's `continue` unconditionally bypasses accumulation.

UNCOVERED by tests: tests/xdart/test_append_skip_before_read.py:44 forces `worker.series_average = False`, so this exact series_average+Append collapse path is untested.  ||  [trigger:CONFIRMED] This is a correctness/dropped-frame bug (not a memory/time-cost bug), fully constructible from the code. The skip-before-read cap that SHOULD gate this per-source-frame does NOT exist for series-average — instead the existing "already processed" gate is what CAUSES the drop.

Mechanism, quoted:
- `_append_output_number` collapses EVERY source frame to output number 1 in series-average mode: image_wrangler_thread.py:760-763 `def _append_output_number(self, img_number): if getattr(self, "series_average", False): return 1`.
- `_should_skip_before_read` skips whenever that output number is already in the disk-primed snapshot: :941-946 `output_img_number = self._append_output_number(img_number)` / `if output_img_number in self._append_skip_snapshot(scan_name): ... return True`.
- The snapshot is primed at run start from the existing target .nxs frame indices when write_mode=='Append' (`_append_skip_enabled` :756-758; `_prime_append_skip_snapshots_for_run` :895-916 loads `scan.load_from_h5(...)` and `_remember_append_skip_snapshot`, which stores `_scan_frame_index_snapshot(scan)` :814/821). So if the target .nxs already holds frame 1, the snapshot for that scan = {1, ...} ⊇ {1}.
- Directory-series read loop, :3008-3054: `while len(self.img_fnames) > 0:` ... :3019 `if self._should_skip_before_read(sname, snumber): continue`. Because `_append_output_number` maps every snumber→1 and 1∈snapshot, EVERY source file hits `continue`. The read at :3022 and the running-mean fold at :3033-3044 are never reached; `n` never increments. Loop exits with `n==0`, returning `(None, None, 1, None, {})` (img_file=None, img_data=None per the :3006 init).

Consequence follows at 5000 frames: the collect loop consumes that as end-of-stream — :1023 `img_data ... = self.get_next_image()`; :1026-1030 `if img_data is None: ... break`. With `pending` empty and `files_processed==0`, the averaged frame the user expected is never dispatched (`_append_series_average_pending` at :1127/1302 is never called) and nothing is written. All 5000 source frames of a series-average Append re-run (or averaging into a .nxs already holding frame 1) are dropped-before-read, producing ZERO output with only an info-level "already processed" skip (`_report_run_skip_summary` :970-971 even suppresses the shortfall warning for series_average). This path is on the normal per-scan directory-series route (not single_img, not viewer-only): `series_average` is a run config flag threaded through the whole thread (:478, :1111, :1450, :2358, :2474). LENS-2 reachability holds: default series-average + Append write_mode + a pre-existing target file is a realistic re-run.

Coverage gap confirmed: tests/xdart/test_append_skip_before_read.py:44 `worker.series_average = False`, so the collapse-to-1 interaction is untested.

Cost framing: not a memory/O(n) cost — this is a full-run data loss; at 5000 source frames the entire acquisition yields no averaged output.
```

</details>

---

### [13] ⚪ LOW · copy-bloat · PLAUSIBLE

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:1300`

**Summary:** Series-Average running mean copies each folded image as np.asarray(value, dtype=float).copy() and computes (current*old_count + incoming)/new_count — full float64 image arithmetic per source frame.

**Failure @ 5000 frames:** In Average-Scan mode each source frame triggers a fresh float64 copy of the ~16 MB image plus a full-array multiply/add/divide (3-4 transient 64 MB arrays per fold). Bounded to ONE running-mean entry (not accumulated), so no unbounded growth — but it is ~4x-upcast full-image arithmetic on every source frame, a steady per-frame CPU/alloc cost that scales with the number of source frames averaged. Not OOM. Fix: accumulate a running sum in-place (out= on the multiply/add) and divide once at emit, avoiding the per-fold .copy() and temporaries.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The mechanism is real but the candidate misstates two specifics. REACHABILITY: only in Average-Scan mode — `series_average` guards the fold (image_wrangler_thread.py:1126 `if series_average:` → :1127 `_append_series_average_pending`; and :1336 `if not ...series_average... return list(pending)`). Per source frame the collect loop folds one image into a single running-mean entry.

REAL per-fold arithmetic (:1289-1292): `return (np.asarray(current, dtype=float) * old_count + np.asarray(incoming, dtype=float)) / new_count` — this allocates ~3 transient float64 arrays per fold. At the stated 16 MB image that is ~48 MB of transient allocation per source frame folded. BOUNDED to one entry: result is assigned to `pending[0]` (:1324), never accumulated — so working set is O(1), NOT O(n). No unbounded retention, no OOM (matches the candidate's own "Not OOM").

REFUTES part 1 — the `.copy()` at :1300 is NOT per-frame. `_copy_average_payload` is only called from the first-fold branch (:1311 `if count <= 0 or not pending:` → :1315/:1317). Every subsequent frame uses `_average_payload` (:1327/:1330), no copy. So `.copy()` is O(1) per averaged scan, not per source frame as the candidate implies.

REFUTES part 2 — the claimed "~4x upcast (uint16→float64)" does NOT happen at this site. `img_data`/`bg_raw` arrive ALREADY float64: `img_data = np.asarray(read_image(self.img_file), dtype=float)` (:2970) and `data = np.asarray(read_image(fname), dtype=float)` (:3022). The `dtype=float` inside `_average_payload`/`_copy_average_payload` is a no-op re-cast on already-float64 data; the real (single) upcast is upstream at read, so counting a 4x quadrupling here double-counts it.

COST AT 5000 FRAMES: negligible memory (one ~16 MB running-mean entry + ~48 MB transient per fold, freed each fold). CPU is O(source_frames_averaged) full-image float64 multiply/add/divide — a steady low per-fold cost, not compounding, not super-linear, not O(n^2). Correctly rated low. Optional micro-opt (in-place `out=` running sum, divide once at emit) is valid but saves only the per-fold temporaries, not a scale killer.  ||  [trigger:PLAUSIBLE] The mechanism is real and quoted exactly. In Series-Average mode every source frame reaches this path: collect loop line 1126-1128 calls `_append_series_average_pending(self, pending, entry, pending_avg_count)` per image read. The fold does full float64 image arithmetic each frame — `_average_payload` (lines 1289-1292): `(np.asarray(current, dtype=float) * old_count + np.asarray(incoming, dtype=float)) / new_count`, and the first-frame seed `_copy_average_payload` (line 1300): `np.asarray(value, dtype=float).copy()`. For a 16 MB uint16/uint32 image the `dtype=float` upcast yields ~64 MB float64, and the multiply/add/divide expression allocates ~3-4 transient ~64 MB arrays per fold. Applied to img_data AND bg_raw (lines 1327, 1330), so up to ~8 transient 64 MB arrays per source frame.

NOT OOM / bounded — confirmed: the pending list is collapsed to exactly ONE running-mean entry. Line 1312 `pending[:] = [( ... )]` seeds one entry; line 1324 `pending[0] = ( ... )` overwrites in place; and `output_img_number = 1 if series_average else img_number` (line 1112) means the whole averaged series produces a single output frame. So there is no per-frame retention and no O(n) accumulation — the cost is purely transient per-fold CPU/alloc that scales with the number of source frames averaged. At 5000 source frames that is 5000 folds × ~8 transient 64 MB float64 arrays ≈ hundreds of MB of transient allocation cumulatively and a steady per-frame arithmetic cost, but bounded working set.

Reachability caveat (why PLAUSIBLE not CONFIRMED): the brief's headline scenario is 5000 DISTINCT traces appended into a live waterfall, which requires series_average=False (line 1112 forces a single output frame #1 when series_average=True). So this exact code is NOT on the 5000-row waterfall route — it only fires when the user explicitly enables Average-Scan mode, which is a non-default config gated by `getattr(self, "series_average", False)` (line 1111, default False). The finding itself is correctly scoped to Average-Scan mode and its claimed consequence (a per-source-frame 4x-upcast CPU/alloc drag, not OOM, bounded to one entry) follows from the code as written. Fix suggestion (in-place `out=` running sum, divide once at emit) is valid: `_average_payload` builds fresh arrays every fold rather than accumulating into a preallocated buffer. Low severity as labeled.
```

</details>

---

### [14] 🟠 HIGH · quadratic · PLAUSIBLE

**Location:** `src/xdart/gui/tabs/static_scan/display_controllers.py:311`

**Summary:** _data_snapshot probes the store for EVERY frame label each tick once the live auto-last selection has grown to Overall (all frames).

**Failure @ 5000 frames:** In Int 1D/2D, when the selection covers the whole scan (`len(selected_ids)==len(all_frame_index)`, which live auto-last reaches — see display_logic.py:429 'live auto-last GROWS the selection every tick'), _candidate_labels returns tuple(all_frame_index) (all N) and _data_snapshot calls resolve_frame_data_for_widget (a real store/hydration probe) for each. That is O(n) probes per timer tick × ~n ticks over the run = O(n^2). At frame 5000 the final ticks each do ~5000 store lookups; cumulative ~12.5M probes over the run, all on the GUI thread → a steadily worsening per-tick stall that makes the last hundreds of frames of a 5000-frame Overall live scan crawl.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] MECHANISM CONFIRMED, magnitude modest. The O(N)-probes-per-render path is real and reachable in the brief's exact default scenario (live Overlay/Waterfall of a growing scan).

Chain verified:
- ScanDisplayController.compute_state (display_controllers.py:368): `if mode in (Mode.INT_1D, Mode.INT_2D) and len(selected_ids) == count and count > 1: all_index, gi = self._all_frame_index(widget)` — materializes the FULL N-label list when the selection covers the whole scan.
- _candidate_labels (display_controllers.py:98-99): `if len(selected_ids) == len(all_frame_index) and len(all_frame_index) > 1: return tuple(all_frame_index)` — returns all N labels.
- _data_snapshot (display_controllers.py:311, 137-160) loops `for label in _label_keys(labels)` calling resolve_frame_data_for_widget once per label for INT_1D, and TWICE (r_2d line 139 + r_raw line 143) for INT_2D. Each probe bottoms out in a lock+dict get: PublicationStore.get (frame_publication.py:764-765) `with self._lock: return self._items.get(label)`. build_payload then re-probes all labels again via _store_first_publication_items (display_controllers.py:206-211) → ~2×N (1D) to ~3×N (2D) lock+dict probes PER RENDER.

Reachability CONFIRMED for Overlay/Waterfall live (the brief's scenario): _render_overlay_full_scan (static_scan_widget.py:4499) sets `self.h5viewer.frame_ids[:] = selected` (all scan.frames.index), so resolve_selection (display_logic.py:855) `overall = (len(frame_ids) == n_all) and (n_all > 1)` is True and stays True every subsequent flush (frame_ids keeps the full set between throttle windows). The 0.5s throttle (static_scan_widget.py:4398-4404) only gates the re-SELECT, not update() itself — the light `_h5viewer_data_changed_now` on throttled ticks still drives displayframe.update() → _live_display_state (display_frame_widget.py:1625) → compute_state → the O(N) snapshot. So every coalesced flush pays O(N) probes.

REFUTATION of the candidate's O(n²)-STALL claim: (1) The candidate's premise that plain auto-last reaches Overall is WRONG for the default single-frame auto-last case — _select_latest selects ONE list item, so _data_changed_now (h5viewer.py:2314-2328) sets frame_ids=[latest] (single, overall=False, O(1) snapshot). Overall requires Overlay/Waterfall's full-scan reselect (line 4499) or explicit show_all (h5viewer.py:1144-1145) — a narrower trigger than "every live tick". (2) Renders are COALESCED at a fixed wall-clock cadence (a few Hz via _pending_update_idx), NOT one per frame, and Overlay full-reselect is throttled to 0.5s — so it is not "~n ticks each doing n probes" as claimed. (3) Each probe is a genuine O(1) dict get under a lock, ~1µs; at frame 5000 that is ~5000 (1D) to ~15000 (2D) probes ≈ 5-15 ms per render — a real, linearly-growing GUI-thread cost that compounds over the run (cumulative O(N²) probe count), but ms-scale, not the "crawl/stall" the candidate asserts, and NOT raw-image/full-2D retention. Cost class = probe/CPU (metadata-sized), not the OOM killer class the review targets. Real and worth fixing (cap _candidate_labels to render_ids or a decimated set before snapshot), but PLAUSIBLE rather than CONFIRMED-stall.  ||  [trigger:REFUTED] The finding's premise — "live auto-last GROWS the selection every tick" to len(selected_ids)==len(all_frame_index) — is factually wrong for widget.frame_ids. The O(n)-per-tick probe branch is gated behind `len(selected_ids) == count and count > 1` (display_controllers.py:368 in ScanDisplayController.compute_state, and _candidate_labels at :98 `if len(selected_ids) == len(all_frame_index) and len(all_frame_index) > 1`). Reaching it requires ALL N list rows to be selected. But during a default live scan the selection is Auto Last, which selects exactly ONE frame: `_select_latest()` (h5viewer.py:999-1010) does `set_current_frame(last_row)` on the single latest row, and `_data_changed_now` (h5viewer.py:2313-2328) sets `self.frame_ids += sorted([str(item.text()) for item in items])` from `listData.selectedItems()` — one item under Auto Last. So `frame_ids` has length 1, `compute_state` takes the else branch `all_index = _FrameIndexCount(count)` (:371), `_candidate_labels` returns `tuple(selected_ids)` = ONE label, and `_data_snapshot` (:137) probes resolve_frame_data_for_widget for that single label => O(1) per tick, O(n) total. This is exactly what `_FrameIndexCount` is designed to guarantee (docstring :106-113: "Auto-last/Single live renders only need len(all_frame_index)... Carrying a length-only object avoids copying thousands of labels on every timer tick"). The all-N branch is reached only via the explicit Show All button (h5viewer.py:1141-1145 `self.frame_ids += self.scan.frames.index`), which is a one-shot click, NOT the default, and is NOT re-latched per tick. Crucially the O(n^2) claim self-contradicts under a GROWING live scan: as new frames arrive, `resolve_selection` (display_logic.py:855 `overall = (len(frame_ids) == n_all) and (n_all > 1)`) reverts overall→False the instant `len(frame_ids) < len(scan.frames.index)`, dropping back to the length-only fast path. Sustained all-N probing therefore only occurs on a FINISHED/static scan with Show All selected, where no per-frame producer timer is driving repeated renders — so the claimed "last hundreds of frames of a 5000-frame live scan crawl" cannot occur. The display_logic.py:429 comment the finding cites ("live auto-last GROWS the selection every tick") refers to the waterfall accumulator's reset_key vs display generation, not to widget.frame_ids growing to N. Magnitude at 5000 frames: actual per-tick cost is 1 store probe (O(1)), total O(n)=~5000 probes over the run, not ~12.5M; no per-tick raw-image or full-2D work here (probes are readiness checks). No compounding O(n^2) on the reachable hot path.
```

</details>

---

### [15] 🟡 MEDIUM · correctness · PLAUSIBLE

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/frame_publication.py:808`

**Summary:** PublicationStore tier-0 _max_items eviction pops next(iter(_items)) with NO persisted check and NO heavy check — it can drop a still-unsaved or still-heavy publication before rehydration is available.

**Failure @ 5000 frames:** Past 512 live frames the tier-0 loop evicts the oldest item outright (pop from _items + drop from heavy/thumb lists). Unlike FrameRecordStore/LiveFrameSeries it does not honor persist-before-evict: if the 1D hydrator has not been registered yet or the frame's row is not yet on disk, get_or_hydrate returns None and that frame becomes undisplayable (a blank/dropped frame in scroll-back) at 5000 frames. Consequence is a correctness/display gap (dropped frame on reload), not OOM — publications are display caches — but it is the one eviction path here that ignores the persisted gate the other two stores enforce.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The candidate's mechanism is real in the code. Tier-0 eviction at frame_publication.py:807-813 has NO persisted/heavy guard:
  `if self._max_items is not None:` / `while len(self._items) > self._max_items:` / `label = next(iter(self._items))` / `self._items.pop(label, None)`
Contrast the two OTHER tiers in the SAME method which downgrade rather than drop: tier-1 (816-822) `self._items[label] = _semilight_publication(publication)` and tier-2 (825-831) `_lightweight_publication`. And unlike FrameRecordStore/LiveFrameSeries, tier-0 never consults a persisted set — it pops the oldest by insertion order outright. `_max_items` = DEFAULT_PUBLICATION_MAX_ITEMS = 512 (line 35/534), so this fires once the store exceeds 512 live frames (i.e. well before frame 5000).

The dropped-frame consequence is reachable: get_or_hydrate (650-673) does `publication = self._items.get(label)` (655) — None for an evicted label — then `if hydrator is None: return publication` (665-666) returns None, and even with a hydrator, `fresh = hydrator(label)` → `if fresh is None: return publication` (671-672) still returns None. The registered hydrator `_rehydrate_publication` (display_data.py:487-537) returns None on a miss: `if lf is None: return None` (508-509) when `_hydrate_frame_from_disk` finds the frame neither writer-resident nor on the `.nxs` yet. So a tier-0-evicted frame that is not-yet-persisted becomes undisplayable (blank/dropped on scroll-back).

Why PLAUSIBLE not CONFIRMED: magnitude/trigger is timing-bound. Tier-0 only evicts frames older than the most-recent 512; by then the LIVE_SAVE_INTERVAL cadence has almost always persisted them to disk (docstring 496-500: 'serves only the writer's resident in-memory frames'), so rehydration normally succeeds and no gap appears. The undisplayable window is the narrow race where a >512-old frame is evicted from `_items` before its `.nxs` row is durable. Cost at 5000 frames is NOT memory (publications are display caches; the heavy raw/2D were already stripped by tier-1 at 64 and tier-2 at 512, so tier-0 evicts metadata-sized lightweight entries — no OOM, no GB growth); it is at most an occasional blank/dropped frame in scroll-back. This is the one eviction path here that ignores the persist-before-evict gate the other two stores enforce, so the fix (gate tier-0 on persisted, or downgrade-not-drop) is warranted, but the code does not prove a guaranteed dropped frame absent the timing window.  ||  [trigger:REFUTED] The tier-0 pop IS persist-agnostic and heavy-agnostic exactly as described — frame_publication.py:808-813: `while len(self._items) > self._max_items: label = next(iter(self._items)); self._items.pop(label, None); self._drop_heavy_label_locked(label); self._drop_thumb_label_locked(label)` — no `_persisted` check and no heavy check, unlike tiers 1/2 (815-832, which only degrade to semilight/lightweight) and unlike LiveFrameSeries.stash (frame_series.py:499-507, persist-before-evict) and FrameRecordStore. Default `_max_items = DEFAULT_PUBLICATION_MAX_ITEMS = 512` (line 35/534) and the store is built no-arg (static_scan_widget.py:388 `PublicationStore()`), so at 5000 live frames tier-0 fires ~4488 times. PATH REACHED.

But the claimed consequence — an "undisplayable (blank/dropped) frame in scroll-back" — does NOT follow, because tier-0 evicts `next(iter(_items))`, i.e. the OLDEST frame (~4500 frames behind the write cursor), which is therefore long past the save cadence and PERSISTED to the .nxs. Recovery of that evicted frame:
(1) The actual waterfall/overlay 1D route hydrates via get_1d_many_or_hydrate (frame_publication.py:675-701) -> _rehydrate_publications_1d, which reads straight from disk: display_data.py:558-561 `with DisplayDataMixin._locked_scan_read(self, scan): results.append(get_1d(scan_file, frame=chunk))`. This runs under the writer-coordinating lock and works DURING an active run — so a tier-0-evicted 1D trace is rebuilt from the .nxs. Recovered.
(2) The single-frame heavy/2D get_or_hydrate (frame_publication.py:650-673) delegates to _hydrate_frame_from_disk, which during an active run serves only resident in-memory frames (display_data.py:444-446 `if getattr(self,'_processing_active',False): return in_mem.get(int(idx))`) — but that mid-run non-rehydration is the DESIGNED behavior for EVERY evicted frame (tiers 1/2 heavy payloads also don't rehydrate mid-run), not a tier-0-specific defect; the frame rehydrates once idle (idle branch reads frames[idx] from the .nxs, display_data.py:456-459).

The candidate's two named triggers also fail: the hydrator IS registered at widget setup before frames stream (display_frame_widget.py:718 `self.publication_store.set_hydrator(self._rehydrate_publication)`), and "row not yet on disk" cannot apply to the tier-0 victim (the oldest, guaranteed-persisted frame). The only real residual is cosmetic: tier-0 drops the thumbnail stub that tiers 1/2 keep, so a mid-live-scan scroll-back >512 frames on the heavy/2D single-frame panel shows nothing until the run goes idle — a minor thumbnail gap on one panel, not the undisplayable/dropped-frame correctness gap as framed. Magnitude at 5000 frames: no OOM (publications are 1D-trace/metadata-sized caches, not raw 16MB images), no data loss (data is on disk and the 1D route rehydrates it), only a transient missing-thumbnail on scroll-back during a live scan.
```

</details>

---

### [16] ⚪ LOW · leak-retention · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xrd_tools/session/frame_record_store.py:378`

**Summary:** FrameRecordStore _max_items (512) tier evicts ONLY persisted labels; if the durable save cadence stalls, thinned (light) records accumulate past 512 unbounded.

**Failure @ 5000 frames:** The tier-2 (_max_items) loop selects next label where _label_persisted_locked is True; if no persisted label exists it returns None and breaks, leaving _records over cap. In live mode records are upserted persisted=False and only marked after QtNexusSink mark_persisted. If the writer stalls or a mode never persists, _records/_source_ids/_persisted_modes grow one entry per frame → at 5000 frames ~5000 thinned records. These are metadata+axes sized (arrays already thinned at the 64 heavy cap), so ~tens of MB, not OOM — but it is a genuine O(n_frames) unbounded growth that the '512' cap does NOT actually enforce under write-stall.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] Cap does NOT exist under write-stall. Tier-2 eviction (frame_record_store.py:378-390): `while len(self._records) > self._max_items: label = next((candidate for candidate in self._records if not self._require_persisted_for_eviction or self._label_persisted_locked(candidate)), None); if label is None: break`. xdart configures it with require_persisted_for_eviction defaulting True (not overridden at image_wrangler_thread.py:2269-2272; default at :155/:168) and max_items=_LIVE_RECORD_STORE_MAX_ITEMS=512 (:116). In live mode records are upserted persisted=False and only marked via QtNexusSink mark_persisted. If the writer stalls (or a mode never persists), the generator yields nothing → next(...)==None → the loop breaks with len(_records) > 512, one thinned record retained per frame → ~5000 entries at frame 5000, unbounded in n_frames. The '512' number is not an enforced ceiling; it is only a trigger to ATTEMPT persisted-only eviction.

The candidate UNDERSTATES magnitude. It assumes overflow records are already array-thinned (tens of MB metadata). But the heavy-thinning tier is the SAME persist-gate: _enforce_bounds_locked tier-1 (:367-376) calls _find_evictable_heavy_label_locked (:340-347) which returns None when nothing heavy is persisted (_label_heavy_payload_persisted_locked, :358-365) → `break` at :372, so under a TOTAL write-stall arrays are NEVER thinned either. In that case every record keeps view.intensity_2d (the full 2D cake) + 1D traces, so it is ~5000 full-2D cakes, not metadata — OOM-class, multi-GB (e.g. n_chi×n_q×8B × 5000). Under a PARTIAL stall (heavy modes persist, some light mode never does) heavy stays capped at 64 and the candidate's ~tens-of-MB metadata figure holds. Either way the '512 cap' is provably not enforced under the write-stall precondition the candidate names. Timing/live-mode dependent (needs writer stall or a non-persisting mode), hence a genuine reachable hazard.  ||  [trigger:PLAUSIBLE] Mechanism confirmed at frame_record_store.py:378-394. The tier-2 (_max_items) eviction loop selects `next((candidate for candidate in self._records if not self._require_persisted_for_eviction or self._label_persisted_locked(candidate)), None)` and `if label is None: break` — so when NO persisted label exists, it breaks leaving `_records` over cap. The cap is real in the live path: image_wrangler_thread.py:2269-2272 constructs `FrameRecordStore(max_items=_LIVE_RECORD_STORE_MAX_ITEMS, max_heavy_items=frame_cap)` with `_LIVE_RECORD_STORE_MAX_ITEMS = 512` (line 116) and NO `require_persisted_for_eviction` arg, so it defaults True (frame_record_store.py:155,168). Live upserts are `persisted=False` and marked only after QtNexusSink mark_persisted (upsert default `persisted: bool = False`, line 196), so under a sustained writer stall the '512' cap does NOT enforce and `_records`/`_source_ids`/`_persisted_modes` grow one entry per frame → ~5000 thinned records at frame 5000.

WHY PLAUSIBLE not CONFIRMED — LENS 2: The path IS on the per-frame live route (upsert→_enforce_bounds_locked runs every frame, line 235), reachable in default config. BUT the unbounded growth only triggers under a DEGRADED state — a sustained write-stall where the sink marks NOTHING persisted; on the normal happy path the save cadence supplies mark_persisted calls (mark_persisted→_enforce_bounds_locked, line 265) so persisted labels exist and eviction proceeds, bounding at 512. Magnitude at 5000 frames is trace/metadata-sized, NOT a killer: raw images and full-2D cakes are already dropped by the INDEPENDENT heavy cap (max_heavy_items=frame_cap≈64, line 367-376 → `_thin_record`), and `_thin_view` (line 76-86) sets intensity_1d/2d/raw/thumbnail=None while keeping only axes + labels + metadata. So ~5000 thinned records ≈ tens of MB (axis grids + scan_info), not OOM. This is genuine O(n_frames) growth the '512' cap fails to bound under write-stall, but the consequence is modest steady-state growth, not crash — matching the 'low' severity. No image/full-2D retention here.
```

</details>

---

### [17] ⚪ LOW · quadratic · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/ewald/scan.py:536`

**Summary:** _save_to_nexus marks the ENTIRE frame index persisted every save (mark_persisted(list(self.frames.index))) instead of only the newly-flushed tail — O(n) work per save → O(n²/interval) over the run. [same root cause also at: src/xdart/modules/ewald/nexus_writer.py:1493]

**Failure @ 5000 frames:** After each durable flush (~every 56 frames) this rebuilds a full list of all N frame indices and set-updates _persisted with all of them. At frame 5000 the last save builds a 5000-int list and does a 5000-element set update; summed over ~90 saves the integer bookkeeping is O(n_frames²/interval). It is cheap ints (no arrays), so not OOM and not a stall, but it is strictly super-linear redundant work on the persist path and only the tail idxs are new — passing the whole index each time is wasted. Correctness is fine (marking already-persisted frames again is idempotent).

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] The candidate is REAL and correctly bounded (its own "quadratic, low" verdict holds).

scan.py:536 (inside `_save_to_nexus`, run every save):
  `mark(list(self.frames.index))`
where `mark = getattr(self.frames, "mark_persisted", None)`. It passes the ENTIRE frame index, not just the newly-flushed tail.

frame_series.py:518-519 proves the per-call cost is O(len(idxs)):
  `with self._cache_lock:`
  `    self._persisted.update(int(i) for i in idxs)`
Building `list(self.frames.index)` is O(N) and the `set.update(int(i) for i in idxs)` iterates and int-coerces all N indices — so each save is O(N_frames_so_far), and only the tail is actually new.

Cadence (magnitude at 5000 frames): FlushPolicy(interval=LIVE_SAVE_INTERVAL, cap=64, margin=8) — the cap−margin pressure bound forces a save at least every ~56 frames (image_wrangler_thread.py:2022-2038, "cap−margin pressure bound"). So ~90 saves over the run. Save k rebuilds a ~(56·k)-element list and set-updates it; summed = Σ 56·k for k=1..90 ≈ O(N²/interval) ≈ ~225k int operations total. The final save alone builds a 5000-int list and does a 5000-element set update.

Cost quantification: pure Python-int bookkeeping (a few hundred KB list transiently, no arrays, no images). NOT OOM, NOT a stall — strictly super-linear REDUNDANT work (re-marking already-persisted frames is idempotent per the set semantics at :519). Only the tail idxs are new each save, so passing `list(self.frames.index)` wastes the redundant portion. The candidate's magnitude, mechanism, and low severity all match the code exactly.  ||  [trigger:PLAUSIBLE] scan.py:534-536 (`_save_to_nexus`, the streaming persist path reached on every durable flush): `mark = getattr(self.frames, "mark_persisted", None); if callable(mark): mark(list(self.frames.index))`. `list(self.frames.index)` materializes the ENTIRE frame index (which grows one int per frame across the run), and mark_persisted (frame_series.py:518-519) does `with self._cache_lock: self._persisted.update(int(i) for i in idxs)` — an O(n) int-coercion + set-update over ALL indices, not just the newly-flushed tail. This path IS on the per-frame live/batch route (it's the inner v2 writer, gated only by save cadence via FlushPolicy, image_wrangler_thread.py:2037), so at 5000 frames it fires periodically (dozens–hundreds of saves depending on interval). At save k covering ~k·interval frames the work is O(k·interval); summed over the run = O(n²/interval). Magnitude at 5000 frames: the final save builds a 5000-element list and does a 5000-element set update; cumulative integer bookkeeping is strictly super-linear. CONSEQUENCE is correctly scoped: pure Python ints, no arrays retained (the raw-image/full-2D killers are absent here) — so NOT OOM and NOT a stall, just redundant super-linear CPU on the persist path where only the tail idxs are ever new. Idempotent, so correctness is fine. The self-classification (quadratic, low) is accurate; magnitude is modest because it's integer work, which is why this is PLAUSIBLE rather than CONFIRMED as a scale killer — the mechanism is real and reachable but the consequence is wasted CPU, not the multi-GB/crash class the review hunts for.
```

</details>

---

### [18] ⚪ LOW · quadratic · PLAUSIBLE

**Location:** `src/xdart/modules/ewald/nexus_writer.py:1500`

**Summary:** On a metadata cursor-miss / reorder / finalize (tail_ids is None), _write_incremental_metadata falls to full _write_scan_metadata/_write_positioners/_write_per_frame_geometry, each reindexing the ENTIRE scan_data DataFrame over all frames — O(n) per save instead of O(new).

**Failure @ 5000 frames:** scan_data is metadata (floats/strings per frame), so transient RAM is modest — this is CPU/IO compounding, not OOM. Normally one-shot (finalize) or rare. But if a mid-scan condition keeps tripping the cursor miss (e.g. out-of-order frames), each of the ~90 saves in a 5000-frame run rewrites the full N-row metadata table, making aggregate metadata work O(n^2) → a growing per-save stall late in the run. Low likelihood on the uniform hot path.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] The mechanism is REAL and the cap is a cursor guard, not an absolute bound. In `_write_incremental_metadata` (nexus_writer.py:1499-1505): `tail_ids = None if finalize else _metadata_tail_ids(...)` and `if tail_ids is None: _write_scan_metadata(...); _write_positioners(...); _write_per_frame_geometry(...)`. The full path is genuinely O(n): `write_scan_metadata` (nexus.py:2347-2372) does `scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)` over ALL frames, then `del entry_grp["scan_data"]` and recreates `frame_index` + every column dataset for the whole table each call.

TRIGGER (reachable but off the uniform hot path): `_metadata_tail_ids` returns None only when (nexus_writer.py:1463-1464) `n > len(ids)` (disk longer than memory) or (1470-1472) `disk_ids != ids[:n]` (disk prefix != memory prefix, i.e. reorder), or `finalize=True` (1499). On ORDERED live acquisition the cheap branch fires: line 1468 `if cursor.metadata == (n, disk_last, memory_sig) and memory_last == disk_last: return ids[n:]`, and upsert_* append only the tail (1511-1528). The design comment (nexus_writer.py:646-647) confirms: "Use tail upserts during ordered acquisition; reconcile the full metadata tables only after reload, reorder, or finalization." Live mode "never passes finalize=True" (640).

MAGNITUDE at 5000 frames: This is CPU/IO compounding, NOT OOM — scan_data is metadata only. `write_scan_metadata` stores float32 numeric columns + UTF-8 string columns (nexus.py:2337-2339, 2362-2372), NO image/2D arrays; a 5000-row table with a handful of columns is a few MB transient. If a mid-scan condition keeps tripping the cursor miss (out-of-order/reorder), each of the ~90 saves (interval ~56) rewrites the full N-row table → aggregate O(n²) rewrite work + full-group delete/recreate churn → a growing per-save stall late in the run. Normally one-shot (finalize only), so low likelihood. Additional per-save O(n) even on the CHEAP path: line 1467 `_index_structure_signature(scan.frames.index, n)` falls to `list(index)[:n]` + `hash(tuple(prefix))` (nexus_writer.py:365-366) unless the Index carries `_structure_version` (362-364) — ~90×O(n) int hashing, cheap in absolute terms but strictly super-linear. No raw-image or full-2D retention on this path.  ||  [trigger:PLAUSIBLE] Mechanism confirmed in src/xdart/modules/ewald/nexus_writer.py. _write_incremental_metadata:1499-1505: `tail_ids = None if finalize else _metadata_tail_ids(...)` then `if tail_ids is None: _write_scan_metadata(...); _write_positioners(...); _write_per_frame_geometry(...)`. The fallback is a FULL rewrite: write_scan_metadata (src/xrd_tools/io/nexus.py:2347-2372) reindexes to ALL frames (`_reindex_scan_data_to_frames(scan_data, frame_indices)`), then `del entry_grp["scan_data"]` and `create_dataset` for `frame_index` + EVERY column over all N rows — O(n) per save. scan.frames.index (frame_series.py:633 `self.index.append(frame.idx)`) accumulates all N ids, so the reindex spans all frames.

Cost class = CPU/IO, NOT OOM: write_scan_metadata:2337-2339,2362-2372 stores numeric columns as float32 and non-numeric as UTF-8 strings — metadata, not the 16MB raw images or full-2D cakes. So transient RAM is a few MB across 5000 frames; the killer here is per-save wall-time, not retention.

Cursor-miss trigger, NOT the uniform hot path: _metadata_tail_ids:1462-1474 returns the cheap tail (`ids[n:]`) whenever `cursor.metadata == (n, disk_last, memory_sig) and memory_last == disk_last`, OR when `disk_ids == ids[:n]`. It returns None only when `n > len(ids)` (1463-1464, frames removed) or `disk_ids != ids[:n]` (1471-1472, reordered/out-of-order prefix), or on `finalize` (1499). On the sequential 5000-frame append (0,1,2,…) the prefix always matches → cheap tail path taken → fallback NOT hit. finalize (one-shot end-of-scan) hits it once (~one O(n) rewrite), harmless.

Consequence at 5000 frames follows ONLY under a recurring cursor-miss (e.g. genuinely out-of-order/reordered frame ids each save): each of the ~90 saves (interval ~56) rewrites the full N-row scan_data → aggregate O(n²) metadata work → a growing per-save stall late in the run. This is state/config-dependent (needs non-monotonic frame ids), not the default uniform path — hence PLAUSIBLE (real, reachable, load/timing-gated), not CONFIRMED. It is not REFUTED: no cap/guard bounds the fallback to the tail, and the reorder path is a real reachable state, not factually wrong.
```

</details>

---

### [19] ⚪ LOW · correctness · PLAUSIBLE

**Location:** `src/xdart/gui/tabs/static_scan/static_scan_widget.py:5636`

**Summary:** new_scan applies the incoming scan's GI/incidence/bai_* config onto self.scan BEFORE the `if not _in_sync: return`, so an out-of-sync new_scan(scanB) burst mutates the GUI scan's geometry while scanA's frames are still rendering.

**Failure @ 5000 frames:** In a multi-scan Image-Directory live run the new_scan signals arrive out of order (the very race the frame-driven boundary was built to tolerate). When new_scan(scanB) fires early and returns via `not _in_sync`, scanB's gi/incidence_motor/bai args have already been written to self.scan; scanA's remaining in-flight frames then pick display axes/GI mode from scanB's config for the ~few frames until the frame-driven boundary rescopes. Wrong-axis / mislabeled display for a handful of transition frames per scan boundary; cosmetic-to-misleading, not OOM. Lower confidence — depends on whether the display axis path reads self.scan.gi for those frames.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:PLAUSIBLE] CONFIRMED the ordering claim: in static_scan_widget.py, GI/incidence/bai config is written to self.scan BEFORE the _in_sync guard. Unconditional pre-guard writes: line 5593-5596 `self.scan.gi = gi` / `self.scan.incidence_motor = incidence_motor` / `single_img` / `series_average`; line 5636-5638 `if self._controls_v2_enabled(): self._controls_v2_ensure_native_int_defaults(); self._controls_v2_apply_gi_config_to_scan()`. The guard is only reached AFTER, at line 5659-5660: `if not _in_sync: return`.

These writes actually mutate self.scan's display-relevant state: `_controls_v2_apply_gi_config_to_scan` (1074-1094) sets `scan.gi`, `scan.gi_config`, `scan.incidence_motor`, `scan.th_mtr`, `scan.sample_orientation`, `scan.tilt_angle`; `_controls_v2_ensure_native_int_defaults` (1286-1295) mutates the live `scan.bai_1d_args`/`bai_2d_args` dicts returned by `_controls_v2_scan_int_args` (1245-1250 return `scan.bai_1d_args, scan.bai_2d_args`), including `a2.setdefault("gi_mode_2d","qip_qoop")` and `a2["unit"]="q_A^-1"`.

RESOLVES the candidate's stated uncertainty ("depends on whether the display axis path reads self.scan.gi"): it DOES, on the normal per-frame live update tick, NOT gated by _in_sync. display_frame_widget.py:3432-3433 `update_binned_view`: `if self.scan.gi: gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d','qip_qoop')` sets the cake axis labels for the currently-displayed frame. Also set_axes at 2962-2966 reads `self.scan.gi`/`bai_1d_args`/`bai_2d_args` to rebuild the plotUnit/imageUnit combos. So scanA's remaining in-flight frames pick up scanB's GI mode/axes for the transition frames until _rescope_frame_panel_to rescopes at the frame-driven boundary.

There is NO guard between the config writes (5593-5638) and the `return` (5660) — nothing refutes it. Consequence: wrong/mislabeled display axes (GI vs non-GI, qip/qoop labels) for a handful of scanA transition frames per out-of-sync scan boundary. Cosmetic-to-misleading correctness bug. NOT a memory/retention issue: zero per-frame growth, zero large-array copy — at 5000 frames there is NO OOM or slowdown from this; it affects only label correctness at the (few-per-scan) transition frames. PLAUSIBLE rather than CONFIRMED because manifestation requires a real out-of-order new_scan burst across scans with DIFFERING GI config plus scanA frames still ticking in the window — reachable per the frame-driven-boundary design (built precisely to tolerate this race) but timing/config-dependent and bounded to a few frames' labels.  ||  [trigger:PLAUSIBLE] CONFIG-BEFORE-GUARD confirmed. In new_scan (static_scan_widget.py) the incoming scan's geometry is written to self.scan BEFORE the in-sync return: line 5593 `self.scan.gi = gi`, 5594 `self.scan.incidence_motor = incidence_motor`; then in the DEFAULT controls-v2 path (XDART_CONTROLS_PANEL_V2 defaults to "1", line 710) lines 5636-5638 run `_controls_v2_apply_gi_config_to_scan()`, which sets `scan.gi = bool(cfg["gi"])`, `scan.gi_config`, `scan.incidence_motor`, `scan.th_mtr`, `scan.sample_orientation`, `scan.tilt_angle` (lines 1074-1094) — all before the guard `if not _in_sync: return` at line 5659-5660. So an out-of-sync new_scan(scanB) that returns via `not _in_sync` has ALREADY mutated the GUI scan's geometry.

CONSEQUENCE-FOLLOWS confirmed on the live per-frame render route. display_data.py get_xydata (def at 1160), the per-frame 2D-cake axis builder, gates the q→2θ conversion at line 1190: `if getattr(self.scan, 'gi', False) or is_gi_2d_units(...)`. Because this is an OR on the mutable self.scan.gi, a wrongly-True scan.gi (flipped to scanB's GI value while scanA's non-GI q/2θ frames are still in flight) short-circuits and returns the radial axis verbatim, SKIPPING the convert_2d_radial q→2θ path at 1199-1205 that a standard non-GI cake needs. Result: wrong/mislabeled radial axis for the handful of scanA transition frames until update_data's frame-driven boundary rescopes. The `or is_gi_2d_units` mitigation (comment 1185-1188, mirrored in display_logic.py:979-989 two_d_kind_from_units) only protects the reverse case (GI units but scan.gi False on reload); it does NOT protect against scan.gi being spuriously True, so the finding's consequence is not neutralized.

Why PLAUSIBLE not CONFIRMED: this is not a retention/OOM path (it is explicitly cosmetic-to-misleading, a few transition frames per scan boundary, self-correcting). The trigger requires (a) a multi-scan Image-Directory live run with the documented out-of-order new_scan burst (comment 5554-5563 describes exactly this race), AND (b) scanA and scanB actually differing in GI mode so the flipped flag changes the axis branch. Both are realistic under default config and the path is reachable on the live route, but magnitude/firing depend on that config difference and the ordering race — load/timing-dependent, hence PLAUSIBLE rather than a quotable-magnitude CONFIRMED.
```

</details>

---

### [20] ⚪ LOW · qt-leak · CONFIRMED

**Location:** `/Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/h5viewer.py:1032`

**Summary:** listData QListWidgetItems grow one per frame for the whole run (addItems(new_tail) appends, never trims mid-scan). [same root cause also at: src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:3010, /Users/vthampy/repos/xrd-tools-integrate/src/xdart/modules/ewald/frame_series.py:461, /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_frame_widget.py:2046, src/xdart/gui/tabs/static_scan/static_scan_widget.py:4038, /Users/vthampy/repos/xrd-tools-integrate/src/xdart/gui/tabs/static_scan/display_frame_widget.py:726]

**Failure @ 5000 frames:** At 5000 frames the frame-list holds 5000 QListWidgetItems (one short label string each). This is genuine per-frame Qt-object accumulation, but each item is tiny (~a few hundred bytes of label + item overhead), so total is only a few MB at 5000 frames — not OOM, just steady low growth. This is the intended selectable-row behavior; only cleared at scan boundary.

<details><summary>Verifier evidence (2-lens)</summary>

```
[real:CONFIRMED] REAL but low-severity, exactly as the candidate frames it. h5viewer.py:1032 `lw.addItems(new_tail)` (and the parallel P4 fast-path at :1079 `self.ui.listData.addItems(new_tail)`) append one QListWidgetItem per new frame every GUI flush. new_tail is built from label STRINGS only: :1027-1030 `new_tail = [str(frame_index[pos]) for pos in range(current_count, len(frame_index))]` — no arrays, no images. There is NO mid-scan trim: the only `listData.clear()` on this update path is scan-boundary, gated on an empty index — :987-989 `if len(frame_index) == 0: self.ui.listData.clear()`. The other clears (:1095-1096 clear+insertItems full rebuild; :1402/1431/1468/1915/2240) are full-rebuild/reload paths, not per-frame trims. So during a single 5000-frame acquisition the widget monotonically accumulates one QListWidgetItem per frame.

Magnitude at 5000 frames: 5000 QListWidgetItems, each holding a short integer-label string ("0".."4999", <=5 chars) plus per-item Qt overhead (~hundreds of bytes to ~1 KB with QListWidgetItem internals) => total on the order of a few MB. This is genuine per-frame Qt-object growth (scenario d) but NOT the killer class: no 16 MB raw image, no full-2D array, no per-frame super-linear work (the append is O(len(tail)), not O(n) — the code explicitly avoids the full rebuild for exactly this reason, comment :1058-1063). Cleared only at scan boundary. Consequence at scale: steady low ~MB growth over the run, not OOM/stall. Severity low, as the candidate states.  ||  [trigger:PLAUSIBLE] The path is real and reachable on the exact 5000-frame live scenario. `update_data` (h5viewer.py:955) is the per-frame GUI-coalesce update path; the live-scan tail fast-path appends items without trimming: line 1032 `lw.addItems(new_tail)` where `new_tail = [str(frame_index[pos]) for pos in range(current_count, len(frame_index))]` (1027-1030), and the equivalent P4 fast path at line 1079 `self.ui.listData.addItems(new_tail)`. These items are plain label STRINGS (`_idxs = [str(i) for i in frame_index]`, line 1041) — no image/2D array data is attached to any QListWidgetItem. The list is only fully cleared at a scan boundary: `if len(frame_index) == 0: self.ui.listData.clear()` (988) or on a full rebuild `self.ui.listData.clear(); self.ui.listData.insertItems(0, _idxs)` (1095-1096). No mid-scan trim exists, so at frame 5000 the widget holds 5000 QListWidgetItems.

Magnitude at 5000 frames: genuine per-frame Qt-object accumulation (scenario d), but each item is only a short label string (~a few hundred bytes + item overhead) = a few MB total, NOT the retained-16MB-raw or full-2D killer class. No cap/del bounds it mid-scan, so REFUTED does not apply; but the consequence is bounded-low, so it is not CONFIRMED-as-OOM. The candidate's own low-severity, non-killer characterization is correct: PLAUSIBLE, steady low growth of ~a few MB, not a crash.
```

</details>

---

## Addendum (2026-07-03) — the 9 GB floor solved (2.3a) + worker-scaling

Follow-up profiling closed out "where is the 9 GB floor." It is **not** the raw
window (the numpy census finds ~0 live arrays at any point) and **not** pyFAI's
base cost (standalone: `load` + `integrate1d(10000)` + `integrate2d(500)` =
**1.96 GB**; split mode nearly irrelevant — no-split 0.97 / bbox 1.96 / full
1.97). It is **per-worker integrator geometry**:

- pyFAI `AzimuthalIntegrator`s are not thread-safe, so
  `_ReductionIntegratorProvider` `copy.deepcopy`s the whole integrator (geometry
  + CSR LUT ≈ 0.8 GB) **per worker thread** — a deliberate thread-safety choice,
  **not** an ADR-0007 violation (ADR-0007 mandates scan-level sharing; the
  per-worker copy is inside that).
- Live-array census at the 5.14 GB setup point: `{'AzimuthalIntegrator': 6,...}`
  — 6 integrators before the first `integrate2d`, climbing as workers spin up.
  `gc.collect` frees none (held C-level → numpy census reads ~0 GB).

### Worker-count scaling (651-frame Eiger, measured)

| workers | peak RSS | time | speedup | note |
|---|---|---|---|---|
| 2 | 6.0 GB | 36.5 s | 1.75× | |
| **4** | **9.1 GB** | **25.0 s** | **2.53×** | throughput knee + production default |
| 8 | 13.7 GB | 26.1 s | 2.50× | +0 speed, +4.6 GB |
| 16 | 19.3 GB | 26.8 s | 2.42× | slower, +10 GB |

Memory ≈ 4 GB base + ~1 GB/worker; **throughput saturates at 4** (the serial
writer thread + GIL-held Python glue are the ceiling, not the 16 cores).

### Correction to the earlier draft

An earlier note claimed "4 workers = full speed AND −3 GB vs the default." Wrong:
it came from the benchmark accidentally running `max_cores=1` (→ the latent
20-worker default pool). The **UI Cores default is 4** (`specUI.py:100`
`min(cpu-1, 4)`, applied in live too), so **production already runs 4 workers ≈
9 GB** — the cap does not lower the common-case floor.

### MEM-3 (implemented) and the deep fix (scoped)

- **MEM-3 guardrail:** `reduction_worker_cap()` (`staging.py`, sharing the MEM-2
  RAM budget) caps the default/fallback pool at the knee (4, floor 2 on <16 GiB),
  maps Cores **honestly** to the pool (fixing the latent `Cores=1 → None → 20`
  bug), env `XDART_REDUCTION_WORKERS`. Effect: prevents the 8/16-worker 14–19 GB
  blowup for zero throughput; the common-case default (4) is unchanged.
- **Deep fix (deferred):** share integrator geometry read-only across workers —
  the only lever that lowers the 4-worker floor at full speed (~3 GB). Declined
  pre-tag on thread-safety / equivalence-spine / pyFAI-coupling risk; scoped in
  `design_reduction_integrator_sharing.md`.
- Cheaper same-floor lever without those risks: finding **[5]** (kill the float64
  ingest upcast) halves the in-flight-scratch half of the per-worker cost.
