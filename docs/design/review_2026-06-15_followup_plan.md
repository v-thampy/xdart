# Implementation plan — review_2026-06-15 follow-ups (the items not yet landed)

**Date:** 2026-06-16 · **For:** a hand-off implementation agent · **Source:** `review_2026-06-15.md`
(see its §0 for what Phase A already landed and verified).

This plan covers **only the review findings that are NOT yet implemented**. It is sequenced by
risk/dependency into waves. Each item is independently shippable unless a dependency is noted.

## 0. LANDED & VERIFIED (2026-06-16, HEAD `d034500`) — supersedes the OPEN snapshot below

Most of this plan has now been implemented (21 commits, `f47790f..d034500`). **Verified by:** the full
suite green (core **1004 passed / 2 skipped**, offscreen xdart **854 passed / 5 skipped**) + a per-commit
adversarial verification pass (**20/21 commits clean; the 1 flagged item is a P3 test-quality nit, not a
code defect**). No P0/P1/P2 issues and **no guardrail violations** (byte-compat, equivalence spine, strict
validators, precision, single-writer thread-safety all intact).

### 0.1 Codex follow-up landing status (2026-06-16, HEAD `61073ef`)

After the `d034500` checkpoint, the remaining offscreen-safe pieces of this plan were taken one step
further. The deliberately live-gated A3/A4 deletion was **not** forced; instead the scan display was
moved toward the publication store while keeping bounded mirrors as a fallback until the live checkpoint.

| Item | Commit | Status |
|---|---|---|
| A3/A4 prep — scan display prefers `PublicationStore` | `26eb7d4` | ✅ scan display no longer OR-merges legacy mirrors into readiness when a publication store is present; Overall aggregate payload can use the on-disk whole-scan aggregate path for evicted rows |
| A3/A4 prep — bound scan-mode mirrors | `9fb96bb` | ✅ `data_1d` and `data_2d` are bounded recent-row mirrors in normal scan modes; viewer modes still keep their Role-B rows unbounded so Image/XYE/NeXus viewer behaviour is not collapsed accidentally |
| QW1 cosmetic leftovers | `4e5433b` | ✅ legacy `update=` argument documented as accepted-but-ignored; stale `add_frame` wording cleaned up |
| N4 P3 threshold-boundary coverage | `4e5433b` | ✅ `_resolve_keep_results("auto")` boundary test added |
| C1 stale RSM scout wording | `4e5433b` | ✅ scout docstring updated from corner-only wording to detector-edge scout wording |
| N3 weak streaming-regression guard | `322b11e` | ✅ Fabio-style `read_image_stack(..., reduce=...)` now has a test proving the stack reader is not materialized |
| Review/design/perf notes | `cb8f1a2` | ✅ June 15 review, follow-up plan, and perf baseline committed |
| Batch-final crash hardening | `33c518f` | ✅ first real batch flush now uses the writer's atomic replace path instead of mutating the skeleton `.nxs` in place; flat detector masks apply by direct flat assignment instead of GUI-thread `concat/unique/unravel` |
| Review follow-up — 2D Sum aggregate + startup guard | `4b3564c` | ✅ evicted Overall `Sum` cakes now call the `sum` aggregate instead of silently rendering an average; `update_scattering_geometry` no longer raises if the GI signal fires before `self.scan` exists |
| Native bus-error fix — avoid LZF stack writes on ARM64 macOS | `61073ef` | ✅ reverses GUI integrated-stack LZF compression and adds the same ARM64 macOS LZF→gzip guard to the shared NeXus writer helper; this mirrors the earlier `ssrl_xrd_tools` `9ff8bf0` bus-error mitigation |
| Fresh review follow-up — aggregate retry, source IDs, norm aliases, docs | this checkpoint | ✅ async Overall aggregate `None` is retryable rather than cached as final; known-vs-missing source IDs no longer merge records; whole-scan norm channels resolve case/alias-insensitively; stale LZF/payload docs refreshed |

**Focused verification after the follow-up:** `tests/xdart/test_frame_publication.py`,
`tests/xdart/test_aggregation_wiring.py`, `tests/xdart/test_gui_modes_end_to_end.py`,
`tests/xdart/test_live_refresh.py`, `tests/xdart/test_display_cross_frame_2d.py`,
`tests/xdart/test_scan_aggregate.py`, `tests/core/test_read.py`,
`tests/core/test_phase_fitting_batch.py`, and `tests/core/test_rsm.py` passed in `xrd_test`
(394 passed / 7 skipped for the broader targeted pass). The crash follow-up additionally passed
`tests/xdart/test_qt_nexus_sink.py` + `tests/xdart/test_frame_publication.py` (79 passed) and the
writer/live-refresh slice `tests/xdart/test_nexus_writer_roundtrip.py`,
`tests/xdart/test_cadence_unified.py`, `tests/xdart/test_live_refresh.py` (317 passed / 1 skipped).
The Sum-aggregate/startup-guard follow-up passed `tests/xdart/test_aggregation_wiring.py` +
`tests/xdart/test_frame_publication.py` (71 passed) and the focused GUI aggregate slice
(`16 passed / 151 deselected`). The native bus-error fix passed
`tests/core/test_nexus.py::TestWriteNexus::test_compression_lzf_uses_safe_filter`,
`tests/core/test_nexus.py::TestWriteNexus::test_compression_lzf_guard_maps_arm_macos_to_fast_gzip`,
`tests/core/test_headless_write_roundtrip.py::test_write_integrated_stack_bulk_then_incremental`,
`tests/xdart/test_nexus_writer_roundtrip.py`, and `tests/xdart/test_qt_nexus_sink.py`
(`81 passed`). The fresh review follow-up passed
`tests/xdart/test_aggregation_wiring.py`, `tests/xdart/test_scan_aggregate.py`,
`tests/xdart/test_frame_publication.py`, and the two LZF guard tests (`91 passed`).

**Still live-gated:** full A3/A4 deletion of Role-A `data_1d`/`data_2d` mirrors. The current state is
a safer pre-live-checkpoint boundary: publication-store-first display, bounded mirrors, and tests for the
disk/tail aggregate path, but the legacy mirrors are still present until the live GUI verifies the next
renderer/data-source flip.

| Item | Commit | Status |
|---|---|---|
| QW1 bai accumulators | `151b516` | ✅ verified (cosmetic: dead `update=` param + garbled docstring `scan.py:358`) |
| QW2 GUI stack compression | `beb5a00` → `61073ef` | ⊘ CLOSED-REVERSED — LZF stack compression reintroduced the known ARM64 macOS h5py/HDF5 bus-error class; GUI writer is intentionally uncompressed, while explicit headless LZF requests are guarded to fast gzip on affected platforms |
| QW3 Eiger native dtype | `1641cd6` | ✅ verified — also *fixes* a latent uint16 saturation-ceiling bug |
| QW4 batch kwargs split + QW5 zero-bg skip | `613feca` | ✅ verified |
| QW6 prefetch metadata cache | `1641cd6` (bundled) | ✅ verified |
| N1 stream durable reductions by default | `a9866cb` + `fcff59e` | ✅ verified |
| N3 stream whole-stack loaders | `0a9aa52` + `322b11e` | ✅ code correct + Fabio/HDF5 streaming-regression guards |
| N4 lightweight phase fits | `c9d6348` + `4e5433b` | ✅ verified + auto-retention boundary test |
| N5 streaming session retains every Frame | `fcff59e` (partial) | ◑ images released; light Frame-shell residual is **by-design** — revisit only if 10k-frame metadata RAM shows up |
| **M1 float32 2D payload** | `d0c2996`→`d034500` | ⊘ **CLOSED-DECLINED** — float32 tried, reverted (strain/peak-fit precision). The transient float64 cake (`single.py:181`/`multi.py:212`) was never the change site; on-disk already downcasts to float32, so **no clean win remains**. Not "pending." |
| M2 reuse corrected images for thumbnails | `3d5b8b4` | ✅ verified |
| I1 cache ProcessedScan indices | `0e44748` | ✅ verified |
| I2 cache image-dir seed scans | `2f8d05e` | ✅ verified |
| I3 lazy flat frame masks | `84e1f6b` | ✅ verified |
| I5 bound image-viewer raw cache | `b070ac4` | ✅ verified |
| I4 Overall double-pass/double-copy | — | ⏳ OPEN — sequence after Wave 5 (unchanged) |
| C1 broaden RSM scout | `3216eb3` + `4e5433b` | ✅ verified bit-exact; stale scout wording fixed |
| C2 reject texture for fixed-q | `6098074` | ✅ verified |
| C3 surface reintegration save failures | `74038ae` | ✅ verified |
| C4 guard session state flags | `b4e9990` | ✅ verified |
| C5 off-thread aggregate file-lock | `cb01b13` (+ wiring) | ✅ verified — reads under `scan.file_lock` + regression test |
| C6 RSM combine_grids streaming | `3216eb3` (bundled) | ✅ verified bit-exact |
| Share Axis #2 + cake current-frame | `5c59077` | ✅ verified (P3: no test for the new multi-frame Single-Overlay cake path) |
| N2 batch submit-per-read (rec. A) | — | ⏳ queued — after Phase A+B (`fix_batch_dispatch_overlap_jun2026.md`) |
| **Wave 5 — A3/A4** (`data_1d`/`data_2d` retirement) | `26eb7d4` + `9fb96bb` partial | ◑ offscreen prep landed; final Role-A deletion remains live-gated |
| **GI-AGG** GI >512-frame live Overall truncation (P2) + Overall e2e test gap | — | ⏳ OPEN — `9fb96bb`'s `data_1d` cap (512) now exposes it; **non-GI is safe** (store-residency routes to the disk aggregate). Fix belongs with the GI displayed-mode instant-switch work; detail in §0.2 |

**Residual P3 follow-ups (none blocking):** multi-frame Single-Overlay cake test (Share Axis commit);
optional extra numeric-equivalence assertion in the combine_grids test. The former N3/N4/C1/QW1 P3s
listed here are closed by `322b11e` / `4e5433b`.

**Still ahead:** finish Wave 5 (A3/A4 live-gated Role-A mirror retirement), the **GI-AGG** P2 (GI >512
live Overall truncation + its e2e test gap — §0.2), I4 (post-Wave-5), N2 (batch rec. A, after Phase A+B),
the residual P3 nits above, and Phase B (out of scope — see below).

### 0.2 Newly tracked (2026-06-16): GI >512-frame live Overall truncation (P2) + Overall e2e test gap

Surfaced by the per-commit verification of `9fb96bb` (bounding `data_1d` `0 → 512`). **Non-GI scans are
safe:** whole-scan Sum/Average/Overall routes on publication-**STORE** residency (heavy cap **64**, not
the 512 `data_1d` cap) → the on-disk `io.aggregate` disk⊕tail path, never the bounded dict. **But
`_whole_scan_aggregate` returns `None` for GI scans** (GI displayed-mode resolution is deferred to the
instant-switch step), so a **GI scan > 512 frames during a live run** falls through to the legacy
`get_frames_int_1d`, which now reads the *capped* `data_1d` and, mid-run, serves only resident frames —
**silently truncating** the Overall Sum/Average. The old unbounded `data_1d (max=0)` masked this; the 512
cap moved the truncation threshold from ∞ to 512.

- **Severity P2** — a documented Phase-5 / GI deferral, but a *new* live truncation that did not exist
  before `9fb96bb` (so worth tracking explicitly, not silently inheriting).
- **Fix (belongs with the GI displayed-mode instant-switch work):** serve a GI whole-scan Overall from the
  primary on-disk stack — or refuse + annotate for a non-primary partial stack — but **never truncate** via
  the capped mirror. Land this with, or before, the Wave-5 Role-A deletion (else GI Overall loses even its
  truncating fallback).
- **Test gap:** add an end-to-end test that drives the **real `displayFrameWidget.update()`** in Overall
  Sum/Average at **N > the store's `max_heavy_items` (64)** with frames absent from **both** the store and
  `data_1d`, asserting the rendered trace equals the full-scan aggregate. It would fail if the routing ever
  reverted to reading the bounded `data_1d`. (The existing `test_aggregation_wiring` tests stub
  `_whole_scan_aggregate` or leave the store fully populated, so they do not exercise the eviction→disk
  path through the real render.)

> **(SUPERSEDED by §0 above — the items below have since landed; the table in this note is the earlier
> pre-implementation grounding snapshot, kept for history.)**
>
> **Revised 2026-06-16 — independent verification pass against HEAD `f47790f`.** Every item below was
> re-grounded in current code (paths/line-anchors all resolve; no ghosts). Three changes from the original:
> (1) **C5 is mostly landed** — the off-thread aggregate read is already serialized via the shared
> `scan.file_lock`; it is downgraded from "ELEVATED, do-first, blocks Wave 5" to "verify + add a
> regression test." (2) **Share Axis #2 is FIXED** in `f47790f` (was listed as open WIP). (3) **Dropped
> review findings folded back in:** N5, C6, I5, QW6 (and I4 deferred to post-Wave-5; D6 pointer-only).
> Status snapshot of every item at HEAD:
>
> | Item | Status | Item | Status |
> |---|---|---|---|
> | QW1 | OPEN | N4 | OPEN |
> | QW2 | OPEN (byte-compat caveat) | N5 *(added)* | OPEN — measure first |
> | QW3 | OPEN | M1 | OPEN |
> | QW4 | OPEN | M2 | OPEN |
> | QW5 | OPEN | I1 | OPEN |
> | QW6 *(added)* | OPEN | I3 | OPEN |
> | N1 | OPEN (highest notebook value) | I5 *(added)* | OPEN |
> | N3 | OPEN | C1, C2, C3, C4 | OPEN |
> | C5 | **DONE bar a test** | C6 *(added)* | OPEN |
> | I4 *(added)* | OPEN — **post-Wave-5** | A3/A4 | not started (live-gated) |

## Scope guardrails (apply to EVERY item — do not violate)
- **No `git push` / publish / tag** — maintainer only. Commit per item with the relevant suite green.
- **Persisted NeXus format is frozen + additive-only.** The byte-compat gate
  (`tests/core/test_v2_record_compat.py`) and schema pins (`tests/core/test_schema_as_code.py`) must
  stay green. Any change that could alter on-disk bytes (notably **QW2** compression) must be checked
  against the gate FIRST and coordinated with the maintainer if the pinned signature moves.
- **Keep the writer/reader validators strict** (`validate_integrated_stack_write`,
  `_require_uniform_axes_1d/2d`, `_select_frames_to_write`). Never relax a validator to fix a bug.
- **The live≡batch≡reload spine** (`tests/xdart/test_gi_batch_real_data.py::test_*_equivalence`) is the
  acceptance gate for anything touching reduction/write/display. A failing spine is a real bug.
- **2D orientation convention** holds: `IntegrationResult2D.intensity` is `(radial, azimuthal)`; saved
  stack / `get_2d` are `(chi, q)`. Don't "tidy" the `from_results` transpose asymmetry.
- Line numbers below are **approximate** (pre-Phase-A); re-locate the symbol before editing.
- Label each fix **regression vs pre-existing** in its commit message.

**A3/A4 (retire `data_1d`/`data_2d`, greenfield 8a/8b) IS now included** — see the dedicated section
after Wave 4. It is greenfield-plan work (not a pure review finding), included per the maintainer's
request; it intersects review items §2.D (ordering gate), D2 (thumbnail copies), D5 (hydrated-raw LRU).
It is **live-gated**, so it is structured as offscreen prep + a live-checkpoint handback.

## Out of scope (owned elsewhere — do NOT touch in this plan)
- **Phase B = ADR-0005 store→session relocation**: the headless `FrameRecord` store in
  `xrd_tools.session` (7a), cadence/eviction into the session (7c), and `PublicationStore`-as-projection
  + shared-metadata (D3). Per the terminal agent's **DECOUPLING UPDATE** in
  `design_store_session_steps7_8_jun2026.md`, this is explicitly **separate from `data_1d` retirement and
  NOT required for it** — it is the bigger/riskier "thin xdart" move, gated behind the A→B go/defer
  checkpoint. So D3 metadata-sharing belongs to Phase B, **not** A3/A4.
- **Share Axis vertical-alignment bug (#2)** — **now FIXED in HEAD `f47790f`** (align-then-lock revert);
  was listed here as open WIP — it is done. Only a live *visual* confirmation remains (cake/1D x-axes line
  up vertically under zoom), which is part of the maintainer's cake-and-axis live pass, not this plan.
- **D1 reintegrate-all RAM** — dormant (buttons hidden), a post-v1 feature; track in the deferred register.
- **D6 chunked error-cleanup vs an already-running worker** (`reduction/core.py` ~1540/1960) — LOW;
  parked in `CC_preship_sweep_deferred_jun2026.md`, not carried here. Pointer only so it isn't lost.
- **Batch submit-per-read (recommendation A)** — already specced in
  `fix_batch_dispatch_overlap_jun2026.md`; do it after Phase A+B. Not re-planned here. **But fold in N1
  below** (the headless-default correction that doc is missing).

---

## Sequencing

| Wave | Items | Theme | Why this order |
|---|---|---|---|
| 0 | QW1–QW6 | Quick wins (small, isolated, no deps) | Immediate value, low risk; clears the easy memory/perf/correctness debt |
| 1 | N1, N3, N4, N5 | Headless / notebook performance | The user's stated priority (notebooks = the headless path); N5 is potentially the biggest single memory win (measure first) |
| 2 | M1, M2 | Memory hot-path (measure-first) | Higher risk (precision/parity) — do after the easy wins, behind the spine |
| 3 | I1–I3, I5 | I/O + GUI responsiveness | Independent; GUI items want a manual check (I5 = standalone viewer-memory leak, Role-B) |
| 4 | C1–C6 | Correctness P2s | Independent, small; **C5 is mostly landed — only its regression test remains** (premise was stale) |
| 5 | A3 → A4 | `data_1d`/`data_2d` retirement (greenfield 8a/8b) | The Phase-A finale; **LIVE-GATED**. Offscreen prep then a live-checkpoint handback to the maintainer. (No longer gated on C5 — its file_lock coord already landed.) |
| post-5 | I4 | Display Overall render perf | Deferred until **after** A3/A4 retargets the display read-paths — see I4 |

---

## Wave 0 — Quick wins

### QW1 — Delete the `bai_1d`/`bai_2d` running-sum accumulators (P1 memory, S)
- **Where:** `src/xdart/modules/ewald/scan.py` — `_accumulate_bai_1d`/`_accumulate_bai_2d` (~462-486),
  their calls in `add_frame` (~443-446), the `bai_1d`/`bai_2d` attrs (~159-160, reset ~175-176);
  `src/xdart/gui/tabs/static_scan/scan_threads.py` resets (~338-340); **update**
  `tests/xdart/test_ewald.py` (~172-211, the only reader).
- **Problem:** a full 2D-slab `IntegrationResult2D.__add__` *every frame* (~10 GB of churned allocation
  on a 651-frame 2D scan; dominant non-pyFAI per-frame cost at 10k). The only consumer is `test_ewald.py`
  — there is **no production reader** (the sibling `overall_raw` accumulator was already removed for this
  reason; whole-scan aggregation is now `io.aggregate`). Verified by grep: `scan.bai_1d`/`scan.bai_2d`
  (the result objects, not `bai_1d_args`) are read only in that test.
- **Fix:** delete the `_accumulate_bai_*` methods + calls + attributes; update/remove the `test_ewald`
  assertions that read them. Leave `bai_1d_args`/`bai_2d_args` (integration config) untouched — they are
  unrelated and used everywhere.
- **Gate:** `pytest tests/xdart/test_ewald.py` + offscreen xdart suite green; spine green.

### QW2 — Compress the GUI writer's integrated 1D/2D stacks (CLOSED-REVERSED)
- **Where:** `src/xdart/modules/ewald/nexus_writer.py` `_commit_integrated_1d` (~882-891) /
  `_commit_integrated_2d` (~958-967) omit the `compression` arg; the headless `NexusSink` writes `lzf`.
- **Original problem:** GUI-written files were larger and slower to reload than headless-written ones.
- **Resolution:** do **not** make the GUI writer use LZF. `beb5a00` did so and later reproduced the
  same native ARM64 macOS h5py/HDF5 bus-error class previously fixed in `ssrl_xrd_tools` commit
  `9ff8bf0`. `61073ef` intentionally returns the GUI writer to uncompressed chunked stacks and adds an
  ARM64 macOS LZF→gzip guard in the shared NeXus writer helper for explicit headless LZF requests.
- **Trade-off:** GUI output is larger than LZF-compressed output, but final batch flush avoids the
  known native crash path and avoids gzip CPU cost on the latency-sensitive GUI save path. If file size
  becomes a release blocker later, evaluate an opt-in gzip policy with real batch timing and crash tests;
  do not re-enable GUI LZF on ARM64 macOS.
- **Gate used:** focused writer tests plus live GUI retry of the previously crashing batch path.

### QW3 — Eiger prefetch: read native dtype, stop widening to int32 (P1 memory, S)
- **Where:** `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py` bulk read (~2253,
  `dtype='int32'`) + the single-frame paths (~2416/2419/2424).
- **Problem:** every Eiger frame is widened to int32 on read — a full-frame copy plus double the queue
  footprint for uint16 detectors (~9 → ~18 MB/frame held in the prefetch queue, plus one extra copy/frame).
- **Fix:** read in the dataset's native dtype (`np.asarray(dset[start:end])`, no `dtype=` override). The
  saturation ceiling / `saturation_pixels` work on uint16/uint32 directly; the single float cast in core
  (`_reduce_frame`) provides subtraction headroom downstream.
- **Gate:** live checkpoint on a real Eiger master (saturation handling unchanged) + spine.

### QW4 — `integrate/batch.py::process_scan` must not forward one `**kwargs` to both 1D and 2D (P2 correctness, S)
- **Where:** `src/xrd_tools/integrate/batch.py` `process_scan` loop (~234-260).
- **Problem:** a single `**kwargs` is passed to both `integrate_1d` and `integrate_2d`, so any 1D-only or
  2D-only pyFAI kwarg (e.g. `npt_azim`) aborts every frame.
- **Fix:** accept/split `kwargs_1d` and `kwargs_2d` (or filter by each call's accepted params).
- **Gate:** `pytest tests/core/test_batch.py` + a new case passing a 2D-only kwarg.

### QW5 — Skip the full-frame background allocation when background is scalar 0 (P2 perf, S)
- **Where:** `src/xrd_tools/reduction/core.py` `_subtract_background` (~2165-2176).
- **Problem:** allocates a full float64 frame even when `background` is scalar 0 (the default "no BG"
  case) — a needless full-frame alloc per frame.
- **Fix:** short-circuit: if `background` is None or a scalar 0, return the input unchanged (no alloc).
- **Gate:** `pytest tests/core/test_reduction*` + spine (output identical).

### QW6 — Eiger prefetch re-reads per-frame metadata inside the bulk loop (P2 perf, S) — *added (review §4, was dropped)*
- **Where:** `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py` ~2276 — the bulk prefetch
  loop calls `read_image_metadata(...)` per frame even though the bulk slice already opened the file.
- **Problem:** a redundant per-frame metadata open/read on the hot prefetch path.
- **Fix:** read the metadata once for the slab (or from the already-open handle) instead of re-opening per
  frame. Same file/path as QW3 — natural to do together.
- **Gate:** spine + a live checkpoint (metadata still correct per frame).

---

## Wave 1 — Headless / notebook performance (user priority)

### N1 — Flip `run_reduction`'s default to streaming for durable sinks + document (P2 score, **highest notebook value**, S)
- **Where:** `src/xrd_tools/reduction/core.py` `run_reduction` (~1467-1481, defaults
  `chunk_size=1, executor=None, execution="chunked"`); also fix the "Deeper win" section of
  `docs/design/fix_batch_dispatch_overlap_jun2026.md`.
- **Problem:** a naive `run_reduction(plan, scan, NexusSink(...))` runs **fully serial** — no
  parallelism, no read‖reduce overlap, and `chunk_size=1` defeats Eiger bulk decompression in
  `NexusStackSource.iter_chunks`. This is the **slowest** path and the default the notebook audience
  hits. The fast streaming path (workers read+decode in parallel via `frame.load_image()`) is opt-in.
  The batch-overlap doc currently mis-states this as "already optimal."
- **Fix:** when a **durable, non-Memory** sink is supplied, auto-select `execution="streaming"` (and a
  default `executor`) — mirroring the existing precedent that auto-selects `retain_products=False` on
  exactly that condition. Keep an explicit caller override. Update the docstring to state the default
  path behavior plainly. Correct the batch-overlap doc's "Deeper win" section to distinguish
  default-serial from opt-in-streaming.
- **Caveat:** behavior must stay output-identical (just faster); streaming is already the
  spine-validated path. Confirm Memory/no-sink callers keep the chunked default (they rely on
  `result.frames`).
- **Gate:** existing reduction + contract tests; **add** a test asserting the default with a durable
  sink runs streaming (overlaps/parallelizes) and produces byte-identical output to the chunked path.

### N3 — Stream the whole-stack headless loaders (P1/P2 memory, M)
- **Where:** `src/xrd_tools/integrate/batch.py::process_scan` (`_collect_frames` ~118-124 reads the
  ENTIRE stack + `astype(float)`); `src/xrd_tools/io/image.py` `read_image_stack(reduce=...)` (~227-269)
  and `read_images_parallel` (~272-303) list-then-stack the whole set (doubling).
- **Problem:** whole-stack residency → OOM on realistic/large scans; `read_image_stack(reduce=...)`
  can't serve as the streaming aggregation primitive the project now wants (cf. `io.aggregate`).
- **Fix:** make `process_scan` iterate frames lazily (read one slab → integrate → write → release),
  mirroring `ReductionSession`'s bounded-memory pattern; cast per-frame, not the whole stack. For
  `read_image_stack(reduce='mean'/'sum')`, fold incrementally (running sum + count) instead of
  materializing, or route callers to `io.aggregate`. **First** enumerate consumers — if `batch.py` is
  legacy vs the `ReductionSession` spine, the minimal fix is per-frame iteration + a deprecation note.
- **Gate:** `pytest tests/core/test_batch.py tests/core/test_image_source.py`; add a large-N
  memory-bounded test (peak RSS or a chunk-count assertion).

### N4 — Lightweight batch phase-fit store (P1 memory, M)
- **Where:** `src/xrd_tools/analysis/fitting/batch.py` `FitResultStore.append` (~172), `fit_sequence`
  (~344), `fit_nexus` (~386); consumer `src/xrd_tools/gui/widgets/batch_phase_fit_viewer.py` (~363).
- **Problem:** a full lmfit `ModelResult` + `PhaseFitter` is retained per frame → unbounded RAM over a
  whole scan; `fit_nexus` also materializes the full `(N, n_q)` stack and N pattern tuples up front.
- **Fix:** add a summary-only mode to `append`/`fit_sequence` (keep `redchi`, `success`,
  `phase_fractions`, lattice params, a small params snapshot; drop the heavy `ModelResult` + fitter once
  the summary is extracted). Default the whole-scan paths (`fit_nexus`, `fit_sequence` over many
  patterns) to lightweight; retain full results only for an explicitly requested subset. Stream
  `fit_nexus`'s stack read (pairs with N3).
- **Gate:** `pytest tests/core/test_phase_fitting.py` + a memory/scale test; confirm the viewer still
  has what it plots (summaries cover the per-frame series).

### N5 — (INVESTIGATE FIRST — potentially the biggest headless-memory item) streaming session retains every `Frame` for the whole scan (P1 memory, M) — *added (review §3, was dropped)*
- **Where:** `src/xrd_tools/reduction/core.py` ~1329 & ~1340 — the streaming session does
  `self.scan._frame_by_index[idx] = frame` for **every** frame, in an unbounded dict on the scan.
- **Problem:** if `frame` still references heavy arrays (`map_raw` / the cake), this retains the entire
  scan in RAM and **defeats the bounded-memory promise of the streaming path** — exactly the path N1 is
  about to make the default, and the one the notebook/headless audience hits at 10k frames.
- **Fix — but MEASURE/CLASSIFY FIRST (do not blindly delete):** confirm whether the retained `Frame`
  holds raw/cake data or only a lightweight handle/results, and whether anything downstream (reload,
  `io.aggregate`, result assembly) depends on the dict being complete. If it's heavy and unneeded after
  persist, release the heavy payload once the frame is written (pairs with persist-before-evict); if a
  handle is needed, keep only an index/light record, not the array-bearing `Frame`. If it turns out to be
  required as-is, document why and bound it.
- **Caveat:** this touches the spine's frame lifecycle — keep the equivalence spine + a bounded-memory
  test green; do not change observable results.
- **Gate:** reduction/contract tests + a large-N peak-RSS (or retained-object-count) assertion proving the
  streaming session no longer grows O(N) in heavy frames.

---

## Wave 2 — Memory hot-path (measure-first; higher risk)

### M1 — Drop the float64 casts on the per-frame hot path (P2 memory, M) — ⊘ CLOSED-DECLINED (2026-06-16)
> **Outcome:** the float32 store was tried (`d0c2996`) and reverted (`d034500`) — strain/peak-fit consume
> the in-memory 2D intensity at float64 and cannot recover precision lost to a float32 store. The transient
> float64 cake at `single.py:181`/`multi.py:212` was never the change site, and the on-disk write already
> downcasts to float32, so there is **no clean memory win left here**. Do not reopen unless a new,
> precision-safe target is identified. Original analysis kept below for context.
- **Where:** `src/xrd_tools/reduction/core.py` `_reduce_frame` (~2056, `image = raw.astype(float)`);
  `src/xrd_tools/integrate/single.py` (~181) + `multi.py` (~212) materialize the 2D cake as float64.
- **Problem:** doubles the working set; the writer downcasts the 2D slab to float32 on save anyway, so
  the float64 cake is transient waste (~8-16 MB/frame).
- **Fix:** **measure first** (numeric parity vs the spine). The safe, high-value target is the **2D
  slab** → float32 end-to-end where pyFAI permits. The per-frame raw cast may need to stay float for
  background-subtraction headroom — verify before changing. Prefer a focused 2D-slab change over a
  blanket float32.
- **Caveat:** pyFAI internals may upcast; the equivalence spine + a numeric-parity test (1D and 2D
  byte/array compare within tolerance) must stay green. If parity moves at all, revert and keep float.
- **Gate:** spine + a 1D/2D numeric-parity test + byte-compat (the saved dtype must not change).

### M2 — Stop recomputing `(map_raw - bg)` for the thumbnail (P2 memory/perf, M)
- **Where:** `src/xdart/modules/ewald/frame.py` `make_thumbnail` (~893-894) builds two fresh full-frame
  float32 arrays to redo a subtraction the reduction core already performed.
- **Problem:** redundant full-frame allocation + work per frame on the worker.
- **Fix:** thread the already-background-subtracted image from `_reduce_frame` to the thumbnail
  downsampler instead of recomputing from `map_raw`/`bg_raw`.
- **Caveat:** preserve the thumbnail's downsample + quantization (the persisted thumbnail dtype/shape
  must not change — byte-compat).
- **Gate:** thumbnail roundtrip tests + byte-compat + spine.

---

## Wave 3 — I/O + GUI responsiveness

### I1 — `ProcessedScan` should not reopen the HDF5 file per access (P2 perf, S/M)
- **Where:** `src/xrd_tools/io/read.py` `ProcessedScan.frames`/`frame_indices`/`__len__` (~650-657, open
  per access) and `iter_chunks`/`load_frame` (~717-733, open per frame — O(N) opens defeat RSM
  streaming).
- **Fix:** cache the lightweight `frame_index` on first read (it already caches metadata). For
  `iter_chunks`, open the file once and stream within the iterator. **Preserve** the documented "holds
  no open file handle" contract for the simple getters (cache the small index array, not a handle).
- **Gate:** `pytest tests/core/test_read.py tests/core/test_n1_source_paths.py` + an RSM streaming test
  asserting one open per `iter_chunks`.

### I2 — Move blocking HDF5/detector reads off the GUI thread (P2 perf, M)
- **Where:** `src/xdart/gui/tabs/static_scan/h5viewer.py` NeXus + Image viewer reads on every selection
  change (~2156); `wranglers/image_wrangler.py` `os.walk` + per-file `os.stat` on every parameter-tree
  change in Image Directory mode (~343).
- **Fix:** route viewer reads through the existing hydration-worker pattern (generation-stamped, like
  `FrameHydrationWorker`/the new `AggregationWorker`); debounce the directory walk (reuse the throttle
  utility) and cache `stat` results.
- **Caveat:** off-thread reads MUST be generation-checked (drop stale results on mode/selection change),
  exactly like the existing workers.
- **Gate:** display/viewer tests; a manual GUI check that scrubbing stays responsive.

### I3 — Avoid materializing a full per-frame boolean mask on the submit path (P2 memory, M)
- **Where:** `src/xdart/modules/reduction.py` `_live_frame_mask_as_bool` (~62) builds a full ~4 MB bool
  detector mask per frame in `frame_from_live_frame`.
- **Fix:** share a single frame-invariant mask across frames where it doesn't change (the common case),
  building the bool view once; only rebuild when the mask genuinely differs per frame (GI per-frame
  masks). Pass a reference, not a fresh copy.
- **Gate:** spine + mask/sentinel tests (`test_sentinel_integration_mask.py`).

### I5 — Image-viewer browse pins full ~18 MB raw arrays the LRU never frees (P2 memory, M) — *added (review §3, was dropped)*
- **Where:** `src/xdart/gui/tabs/static_scan/h5viewer.py` ~1916-1987 — browsing the Image Viewer inserts
  full raw arrays into `data_1d` that the hydrated-raw LRU does not evict.
- **Problem:** scrubbing a long scan in the viewer grows RAM unbounded (~18 MB/raw frame retained).
- **Independent of A3/A4:** this is the **Role-B** (viewer) `data_1d` usage that A3/A4 explicitly KEEPS —
  so it is a standalone viewer-memory leak to fix on its own, not part of the Role-A retirement.
- **Fix:** bound the viewer's retained raw frames (apply/extend the hydrated-raw LRU to these inserts, or
  cap + evict on browse), keeping only the displayed frame(s) resident.
- **Gate:** a viewer-browse memory test (scrub N frames, assert bounded retained raw) + a manual GUI check.

### I4 — (SEQUENCE AFTER WAVE 5) Overall render does two O(N) passes + double-copies each 2D cake (P2 perf/memory, M) — *added (review §4, was dropped)*
- **Where:** `src/xdart/.../display_controllers.py` `compute_state` (~194, two full store/scan passes per
  update); `src/xdart/.../display_data.py` ~545 & ~873 (each ~8-16 MB 2D cake copied twice per render).
- **Problem:** per-update GUI cost grows O(N) with the scan and churns a full cake-sized copy each render.
- **Fix:** single-pass the state computation; render the cake from the existing array (view/in-place)
  rather than two copies.
- **⚠️ DEPENDENCY — do this AFTER Wave 5 (A3/A4), not in Wave 3.** A3/A4 retargets exactly these Role-A
  display read-paths off `data_1d`/`data_2d`; optimizing them first would be rewritten work. After A3/A4
  settles, re-confirm the hot path still exists, then fix. (Listed here next to its sibling I-items for
  locality; its true slot is post-Wave-5.)
- **Gate:** display/viewer tests + a render-pass-count / copy-count assertion; spine.

---

## Wave 4 — Correctness P2s

### C5 — (REVISED — mostly landed; residual is a test) Off-thread aggregate read vs the live writer's file lock (P2 concurrency, S)
- **STATUS — the original premise is STALE; re-verified against HEAD `f47790f`.** The off-thread
  aggregate read **is already serialized against the writer.** `whole_scan_aggregate_1d/2d`
  (`src/xdart/modules/scan_aggregate.py` ~201/220) wraps the read in `with _file_lock(scan):`, which
  resolves to `scan.file_lock` — the SAME `Condition` the writer's `save_to_nexus` holds (`ewald/scan.py`
  ~511), unified because the wrangler creates `LiveScan(..., file_lock=self.file_lock)`
  (`image_wrangler_thread.py` ~2563, the "J2" unification). The torn-read correctness gap is **closed**.
  This is **no longer a hard prerequisite for Wave 5** — ignore the "do C5 first" gating below.
- **What's left (the only residual) — lock the behavior with a test** (none currently exercises
  read-during-write):
  1. Concurrency test: aggregate (1D **and** 2D) while a writer thread holds/append-cycles the file under
     the shared lock — assert no exception and a correct result (never the silent `result=None` fallback).
  2. Boundary test for the one genuine race nuance: `_unflushed_tail` snapshots the in-memory tail
     *before* acquiring `file_lock`, so a frame can flush+evict between snapshot and read. `io.aggregate`
     already dedups disk-vs-tail **by label** (verified Round-26), so it's handled, not a torn read — but
     drive exactly that interleave (snapshot tail → flush+evict that label → read) and assert the frame is
     counted **exactly once**.
- **Do NOT "fix" it two tempting-but-wrong ways:** (a) SWMR — the headless writer's SWMR is
  advertised-but-non-functional (`io/nexus.py` ~783) and the xdart writer has none; the `file_lock`
  discipline is the real mechanism and it's already in place. (b) An `H5FilePool` pause/resume bracket —
  unnecessary (same lock, fully serialized) unless a test proves the lock insufficient (it isn't).
- **Gate:** the two tests above green. No separate live checkpoint needed for C5 (the shared lock
  guarantees it); the read path is re-exercised under the Wave-5 A3/A4 live checkpoint regardless.

### C1 — RSM corner-only scout can under-bound the q grid (P2, S)
- **Where:** `src/xrd_tools/rsm/gridding.py` `_corner_pixel_q` (~128).
- **Problem:** sampling only detector corners can under-bound the q range; out-of-range bins are then
  silently dropped.
- **Fix:** sample edge midpoints (or a denser edge set), not just the 4 corners, when establishing the
  grid extent; or expand the bound with a small margin and document it.
- **Gate:** `pytest tests/core/test_rsm.py tests/core/test_streaming_gridder.py` + a case with a curved-q
  geometry where corners under-bound.

### C6 — RSM `combine_grids` builds a ~0.5 GB transient dense meshgrid (P2 memory, S) — *added (review §3, was dropped)*
- **Where:** `src/xrd_tools/rsm/gridding.py` ~595 `H, K, L = np.meshgrid(h, k, l, indexing="ij")`.
- **Problem:** materializes three full dense 3-D coordinate arrays at once (~0.5 GB on a realistic grid) —
  a transient memory spike on the RSM path.
- **Fix:** use `np.meshgrid(..., sparse=True)` (broadcasting views) or compute per-axis without the dense
  product where the consumer allows; densify only the axis actually needed.
- **Gate:** `pytest tests/core/test_rsm.py tests/core/test_streaming_gridder.py` + a peak-RSS check on a
  representative grid. (Same file as C1 — fold the RSM gridding work together.)

### C2 — March-Dollase texture uses a placeholder unit-cubic metric tensor (P2, S)
- **Where:** `src/xrd_tools/analysis/fitting/phase_fitting.py` `_kernel` texture branch (~625-629).
- **Problem:** a structure-less (fixed-q) phase has no real metric tensor, so the texture correction uses
  a placeholder unit-cubic tensor → wrong corrections.
- **Fix:** use the phase's real metric tensor when available; if the phase is structure-less, refuse
  texture (raise/flag) rather than silently applying a wrong correction.
- **Gate:** `pytest tests/core/test_phase_fitting.py` + a case asserting texture on a structure-less
  phase is refused/correct.

### C3 — Surface reintegration write failure instead of swallowing it (P2, S)
- **Where:** `src/xdart/gui/tabs/static_scan/scan_threads.py` `_close_reduction_session` (~120-133) sets
  `self._reduction_write_error = exc` — a flag that is never read.
- **Problem:** a reintegration write failure on `finish()` is silently lost; the user believes the save
  succeeded.
- **Fix:** surface the recorded error (status label / error signal), mirroring how the streaming run path
  reports write failures.
- **Gate:** a test that injects a write failure on `finish()` and asserts it reaches the user-facing
  channel.

### C4 — Guard the streaming `_failure`/`_cancelled` cross-thread flag reads (P2, S)
- **Where:** `src/xrd_tools/reduction/core.py` — the writer thread sets `self._failure`/`self._cancelled`
  (~1012) read by the `submit()` thread without a barrier/lock.
- **Problem:** low-probability stale read on a relaxed-memory platform (benign on CPython today).
- **Fix:** read/write these under the session's existing lock (cheap), or document the CPython-GIL
  assumption explicitly if a lock is deemed unnecessary.
- **Gate:** contract tests (`tests/core/test_contracts.py`) + the streaming/abort tests.

---

---

## Wave 5 — A3/A4: retire Role-A `data_1d`/`data_2d` (greenfield 8a/8b) — LIVE-GATED, L

**Authoritative spec:** the **DECOUPLING UPDATE** in
`docs/design/design_store_session_steps7_8_jun2026.md` (currently uncommitted working-tree edit) — read it
first; it supersedes the original "7+8 coupled" framing. This is **Phase A's finale**, NOT the
session-store relocation (that is Phase B, out of scope).

**What this is / isn't.** A3/A4 deletes the legacy *scan-integration cache* (`data_1d` unbounded,
`data_2d` cap-40) so the publication payload + the on-disk aggregate are the sole 1D/2D integration
source. It does **not** build the headless `FrameRecord` store or make `PublicationStore` a projection —
that is Phase B (7a/7c/D3). The capability A3/A4 depends on already shipped in Phase A.

**Prerequisites (landed — re-verify green before starting):**
- A1 (`io.aggregate` + the xdart `scan_aggregate`/`AggregationWorker` wiring) — whole-scan Sum/Average
  now comes from disk⊕tail, so it no longer needs `data_1d` as the unbounded backstop. ✅
- A2 (`get_or_hydrate` rehydrates tier-1 thumbnail-only frames) — scroll-back to an evicted frame
  rehydrates from disk; this is what makes deleting the unbounded `data_1d` safe. ✅
- **C5 (already landed — NOT a blocker):** the off-thread aggregate read is already coordinated with the
  writer via the shared `scan.file_lock` (re-verified at HEAD — see the revised C5). No file-lock work is
  needed before the flip. The only C5 residual is its regression test, which **should be green before the
  A3/A4 live checkpoint** since A3/A4 leans harder on the concurrent read path — but it does not gate the
  offscreen prep.

### The central discipline — classify EVERY site before touching it
`data_1d`/`data_2d` serve **two roles**; the deletion surface is only Role-A (≈ a third of the 226 raw
refs), not the whole thing:
- **Role-A — scan-integration cache (RETARGET → DELETE):** the integration-mode 1D/2D display, whole-scan
  Sum/Average/Overall, the raw-2D panel `map_raw`, TIFF export, the raw-panel mask, the image preview, and
  `_snapshot_data`/availability. Writers: `frame.copy_for_display(...)` inserts at
  `scan_threads.py:248-262` & `:660-662`, `static_scan_widget.py:723-733`,
  `image_wrangler_thread.py:1953-1956`; the dict decls at `static_scan_widget.py:329-330`. Consumers:
  `display_data.get_frames_*`/`_snapshot_data`, `display_controllers` (`map_raw` source ~133),
  `display_plot`, `metadata`, export.
- **Role-B — viewer-mode arrays (KEEP):** Image/XYE/NeXus viewer plotting in `h5viewer.py` (the
  per-file sequential-index inserts ~1159/1178 and `_ViewerRows`). The original design's "keep
  `_ViewerRows`" *understated* this — the decoupling update is explicit that all viewer-mode `data_1d`/
  `data_2d` usage stays. **Do not delete Role-B.**
- **`hydrated_raw.py` (D5 LRU rides on `data_2d`):** greenfield 8b deletes it, but only once the raw-2D
  panel's window is sourced from the store/`get_or_hydrate` (Role-A retarget). Confirm the raw panel
  renders from the store before removing the LRU.

### A3 — retarget + flip (greenfield 8a)
1. Confirm A2's scroll-back-to-evicted path: a frame past the heavy bound shows its tier-1 thumbnail with
   no GUI stall, then rehydrates full raw off-thread. Add a **scroll-back stall test** if one doesn't
   exist (100-frame scan, small cap, scroll to an evicted frame, assert thumbnail-then-rehydrate, no
   blocking h5 read on the GUI thread).
2. Retarget Role-A reads: integration-1D draw + Sum/Average/Overall → publication payload + `io.aggregate`
   (largely done in A1 — finish any stragglers); raw-2D `map_raw`/preview/export/`_snapshot_data` → the
   store (`get_or_hydrate`) instead of `data_2d`.
3. **8a:** flip the scan-1D draw to **payload-only**; remove the legacy `update_plot` fallback. Any
   remaining Role-A window is bounded by the store, never by `data_1d(max=0)`.

### A4 — delete (greenfield 8b)
4. Stop the Role-A writers (the `copy_for_display` inserts). `copy_for_display` is also the **D2**
   thumbnail-copy source — if it has no Role-B consumer left, deleting it closes **D2** (the ≤256² float32
   thumbnail per 1D entry). Confirm the raw preview/thumbnail now sources from the store/payload first.
5. Delete the Role-A `data_1d`/`data_2d` dicts + `hydrated_raw.py`; keep Role-B (h5viewer viewer arrays +
   `_ViewerRows`).
6. **Done-test (the greenfield "done"):** no "keep both stores in sync"-style comments remain; Role-A
   `data_1d`/`data_2d` identifiers are gone from the integration display layer (grep-clean); Role-B intact.

### Gates (per the design + review §2.D)
- Spine + byte-compat green **per step**.
- The **>64-frame aggregation test** (landed) must stay green **after** the `data_1d` backstop is gone —
  this is the Round-12 regression gate and the entire point of the deletion. **Do not delete `data_1d`
  until it (and the scroll-back test) are green** (review §2.D ordering gate — still binding).
- **LIVE CHECKPOINT (maintainer):** the design marks 8a/8b + scroll-back live-gated. The implementing
  agent does the **offscreen-gatable prep** (retarget, deletion mechanics, snapshot/display/scroll-back
  tests green) and then **hands back to the maintainer** for a live session — real QThread teardown,
  pause/disk races, scroll-back latency on a long real scan — before A3/A4 is declared done. Do not
  self-certify the live behavior.

### Traps
- Deleting Role-B (viewer modes) — classify before deleting; the original design understated Role-B.
- Deleting `data_1d` before the >64 aggregation + scroll-back tests are green (re-opens Round-12).
- **GI Overall truncation (GI-AGG, §0.2):** a GI scan >512 frames *already* truncates a live Overall
  Sum/Average via the capped `data_1d` (non-GI is safe). Restoring full GI coverage (GI instant-switch)
  must land with — or before — deleting the Role-A mirror, or GI Overall loses its only (currently
  truncating) fallback entirely.
- Shipping the A3/A4 live checkpoint without C5's regression test — the read-during-write path is
  lock-protected (already serialized via `scan.file_lock`) but currently **unproven by any test**; land the
  C5 test first so the concurrent path A3/A4 leans on is covered. (The old "skipping C5 races the writer"
  trap is obsolete — the lock is in place.)
- Removing `copy_for_display`/`hydrated_raw` before the raw panel + thumbnail are confirmed sourcing from
  the store.

**Effort:** L. **Depends on:** A1 ✅, A2 ✅; C5's file_lock coordination already landed (only its test is
owed, before the live checkpoint). Phase B (7a/7c/D3) NOT required and NOT in scope.

---

## Notes for the reviewing maintainer
- **Biggest single win for the notebook audience is N1** (small change, scored P2) — it makes the default
  headless path stop being serial.
- **C5's premise turned out stale** (re-verified at HEAD `f47790f`): the off-thread aggregate read is
  already serialized against the writer via the shared `scan.file_lock`, so it is **not** a new risk and
  **not** a hard prerequisite for Wave 5. Its only residual is a regression test for the read-during-write
  path, which should be green before the A3/A4 live checkpoint. (This was the biggest correction to the
  original plan — it was written before the `af4a220`/`940b896` commits added the lock wrapping.)
- **Several still-open review findings the original plan had dropped are now folded in:** N5 (streaming
  session retains every `Frame` — potentially the biggest headless-memory win, measure first), C6 (RSM
  `combine_grids` dense meshgrid), I5 (viewer raw-pin leak, Role-B/independent), QW6 (prefetch metadata
  re-read), and I4 (Overall-render perf — explicitly deferred to **after** A3/A4 since it touches the
  read-paths A3/A4 rewrites). D6 is a pointer-only (parked in the deferred register).
- **A3/A4 is greenfield work, not a pure review finding**, and is **live-gated**: the implementing agent
  should land the offscreen prep + tests, then hand back to you for the live session. If a single agent is
  driving both this plan and the store→session track, A3/A4 here and Phase B there must not both edit the
  `data_1d`/`data_2d` surface concurrently — sequence A3/A4 (delete Role-A) before any Phase-B projection
  work.
- QW2 (compression) and M1/M2 (dtypes/thumbnail) all touch the byte-compat gate — treat the pinned
  signature as the source of truth and coordinate before re-baselining.
