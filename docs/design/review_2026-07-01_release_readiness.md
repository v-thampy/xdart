# Release-readiness review — v1.0.0 (2026-07-01)

**Scope:** pre-release review of `feature/controls-panel-v2` before merging → `feature/geometry`
→ `main` → tagging **v1.0.0**. Two isolated code-running agents (memory/responsiveness; bugs/
quick-wins) + a git-state check. Focus: release blockers + **low-hanging fruit** (hours-to-a-day,
not weeks). All effort tags: **S ≤ 2 h · M ≤ 1 day**.

**Git state.** `feature/controls-panel-v2` is **5 ahead / 0 behind `feature/geometry`** (clean
merge), **234 ahead of `main` with `main` 2 ahead** (two `main`-only commits — see BLOCKER-1),
version `1.0.0`, `MIGRATION.md` modified-uncommitted (expected — doc stays local). The recent log
is heavy active bug-fixing (generic-detector pixel size, geometry restore, Make-Mask freezes,
controls-refresh) — all **verified complete + regression-free** (see bottom).

> **Implementation note (2026-07-01):** the follow-up release-hardening chunk on
> `feature/controls-panel-v2` addressed B-1, S-1, S-2, M-1, M-2, M-3, N-1, and
> N-3; S-3 was verified with chunked offscreen GUI runs and the known exit-139
> retry; N-2 remains deferred because it intentionally re-pins byte-signature
> fixtures.

---

## BLOCKERS (fix before release)

**B-1 — Bring the two `main`-only CI commits through the merge; the branch is RED without them. [S]**
`git log HEAD..main` = `abc5dd24` + `d07a7886` (both pure CI/test, no product code):
- `abc5dd24` "CI: mask entry/reduction/host in the byte-compat gate" — **this is why `tests/core`
  and `scripts/release.py check` are currently FAILING on this branch.** The branch's
  `tests/core/h5sig.py` masks `"hostname"`, but the reduction record writes leaf `entry/reduction/host`
  (a per-machine hash), so `test_v2_record_compat.py::test_v2_record_content_identical_to_pre6a`
  pins *this machine's* host into the fixture and fails on every other host. It is the **only** core
  failure (`1386 passed, 2 skipped` with it deselected). `abc5dd24` adds `"host"` to
  `VOLATILE_LEAVES` and fixes both.
- `d07a7886` "CI: fix xdart test isolation + retry offscreen segfault" — completes GUI mock scans +
  adds the exit-139 retry to `pr.yml`/`nightly.yml` (this branch's workflows lack the retry, so the
  offscreen job is more fragile here).
- **Action:** the `geometry→main` merge must include both. Verify post-merge on a fresh checkout:
  `python scripts/release.py check` green + `pytest tests/core` green.

---

## SHOULD-FIX BEFORE RELEASE

**S-1 — [major] Chunked-PARALLEL reduction drops the strict policy (D7 default-loud defeated on the executor path). [S]**
`src/xrd_tools/reduction/core.py:1483-1499` — the serial and streaming branches pass `strict=self.strict`,
but the parallel `self._worker.submit(_reduce_frame, …)` stops its positional args at
`self._warned_monitor_keys` and never passes the keyword-only `strict=`. So `execution="chunked"` +
executor runs every frame with `strict=None` (graceful) and **silently persists un-normalized data**
where chunked-serial correctly raises `MissingNormalizationError` (runtime-reproduced). Headless-only
(xdart always passes graceful; durable sinks auto-select streaming), but it contradicts the D7 contract
+ MIGRATION.md item 11. **Fix:** add `strict=self.strict,` to the `submit(...)` call; add an
`executor=True/2` variant to `tests/core/test_strictness.py:151` (today only serial-chunked is tested).

**S-2 — [responsiveness, borderline] Full-directory `rglob` + `sorted()` on the throttled refresh path. [S]**
`static_scan_widget.py:2838-2843` (`_controls_v2_first_metadata_file`): for an Image-Directory source
with a metadata ext, every V2 refresh (250 ms throttle, while `img_file` is empty during initial
config) walks the whole tree and `sorted()`s it before returning the first match — a multi-hundred-ms-
to-seconds GUI stall per keystroke on a big data dir (the common "point xdart at a directory" flow).
**Fix:** replace `sorted(candidates)` with an unsorted `next(p for … if match)` (first hit, no full
sort) and cache the probe on `(img_dir, img_ext, filter, include_subdir)` like the metadata read already is.

**S-3 — [process] Confirm the `xdart` offscreen job is green at tip. [S]**
`tests/xdart/test_live_refresh.py`'s "~12 in-file failures, pass individually" is a **test-mock artifact,
not a product bug** (viewer-mode `SimpleNamespace` stub drift — same class `d07a7886` fixed in
`test_gui_modes_end_to_end.py`, "no production code changed"). But CI runs the whole dir with no retry
on this branch, so on a `[gui,dev]` machine run the full file offscreen; if red, complete the ~4 missing
mock hosts (mechanical). Blocked from confirming here (no Qt in the sandbox).

---

## QUICK WINS (low-hanging fruit — do opportunistically with the above)

- **M-1 — [memory, ~144 MB] Drop the per-frame `np.zeros_like` `bg_raw`. [S]** `h5viewer.py:1949,2016`
  allocate a full-size zero array (~18 MB/Eiger) as `bg_raw` for every Image-Viewer/single-frame row,
  but every consumer treats scalar `0`/`None` as "no background" (`display_data.py:265,466`,
  `display_controllers.py:213`). Replace with `'bg_raw': 0` → up to ~144 MB saved (cap-8 LRU). Safe/isolated.
- **M-2 — [memory, trivial] Delete the dead `_raw_cache_order`.** `h5viewer.py:71,679-680` — assigned/
  cleared, never appended/read. Removes the confusion the D₁ item calls out.
- **M-3 — [memory, corner] Cap `data_1d` in viewer modes. [S]** `static_scan_widget.py:5326` sets the cap
  to `None` (unbounded) for Image/XYE/NeXus viewers; entries are light markers, but a huge NeXus/XYE set
  grows one row/frame unbounded. Cap at a large finite value.
- **N-1 — [correctness] `write_rsm` lacks the intensity-shape guard its twin has. [S]** `nexus.py:2018-2021`
  writes `volume.intensity` + h/k/l with no `shape == (len h, len k, len l)` check (`write_stitched`
  enforces the cake contract at `:1973`). A transposed RSM volume round-trips silently wrong. Mirror the guard.
- **N-2 — [gate] Pin compression/chunks/maxshape in the byte-compat signature. [S]** `tests/core/h5sig.py`
  excludes filters; stitch/rsm byte-identity is clean *today* but a future compression change wouldn't be
  caught. Add the filter fields and re-pin.
- **N-3 — [doc] MIGRATION.md / release-note wording: only the *reduction* path is loud-by-default. [S]** The
  one reader with a strict seam (`load_processed_raw_or_thumbnail`) *defaults graceful* by design (it backs
  the viewer). MIGRATION.md item 11 is correct; just don't let any note claim "readers raise."

---

## DEFERRED (post-v1 — known/big, not blockers)

- **R-2 — [responsiveness, M] Synchronous source-open / first-frame / HDF5 probe on the GUI thread**
  (`setup()` → `os.walk`/`read_image_metadata`/`read_nexus`). Stalls on a slow/huge master file; one-shot
  per selection. The async `ScanSourceWidget` (`_start_async_probe` + `ThreadPoolExecutor`) already exists
  but is mounted only in the ROI plot dialog (`async_probe=False`) — wire it into the production Source card.
- **Placeholder-geometry root cause (M).** The `dist<=0` guard (`c41ec287`) fails loud correctly on both
  Run and Reintegrate; whether the *current writer* ever persists a placeholder for valid data is unconfirmed
  (files deleted before repro). Follow-up investigation.
- **D₁ (dual `data_2d`/hydrated-raw lifecycle) + D₃ (dormant `FrameRecordStore` one-store)** — the remediation
  plan's big refactors; **bounded today** (cap-8 hydrated-raw LRU + cap-64 persist-before-evict), not leaks.
- Panel-v2 finish (native pages / retire `ParameterTree`) + F-3 (GI corrections → Int) — the forward roadmap.

---

## VERIFIED OK (reassuring for release)
- The recent reduction/geometry fixes are **complete + regression-free**: `131e8be4` (worker-thread
  `deepcopy` of the base AI keeps generic-detector pixel size), `980bf9fe` (reuse the restored pixel-bearing
  integrator, keyed on poni identity), `c41ec287` (`dist<=0` **fails loud on both** Run + Reintegrate — no
  silent NaN). Make-Mask chain (`6c66aaaa`/`c0aba374`/`f18a6afd`) — detached subprocess + daemon waiter, no
  GUI-thread block, no zombie/leak.
- **D₂ persist-before-evict** invariant holds (`flush_serial_tail` saves before resetting the counter;
  `FlushPolicy` hard `cap−margin` bound). **C** schema-helper routing of `write_stitched`/`write_rsm` is
  byte-identical (incl. filters). **E** sink-contract harness green (10). **B** StrictPolicy default-loud with
  the GUI opting graceful at all 4 sites.
- Memory caches bounded (data_2d cap-40 but hydrated-raw LRU nulls beyond 8; LiveFrameSeries cap-64
  persist-before-evict; `_published_frames` ~16). Batch V2 refresh **suppressed to a single end-of-run** —
  no per-frame panel stutter. `set_state` defers a full rebuild until a focused editor commits. Timers are
  single-shot, no spin; FormRows `deleteLater`'d.
- `pyproject` 1.0.0 metadata sound (requires-python ≥3.11, `[gui]` extras match the PySide6 pin, version
  preflight passes). `tests/core` = **1386 pass / 2 skip** (once B-1 lands).

---

## Recommended release sequence
1. Land **B-1** (merge geometry→main *with* the 2 main CI commits) — turns CI/preflight green. This is the
   gate; nothing else can be validated until it's in.
2. Land **S-1** (one-line `strict=` fix + test) — the only real headless correctness hole.
3. Opportunistically land the **S** quick wins in one small commit: **S-2** (rglob), **M-1** (bg_raw),
   **M-2/M-3**, **N-1/N-2/N-3**. All isolated, all ≤ 2 h, none risk the byte-compat/spine gates.
4. **S-3**: run the full `xdart` offscreen suite on a `[gui,dev]` box; fix mock hosts if red.
5. Re-run `scripts/release.py check` + `pytest tests/core` + offscreen `tests/xdart` on a fresh checkout →
   tag v1.0.0.
