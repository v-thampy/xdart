# Deep Review — ssrl_xrd_tools + xdart (`refactor/architecture-v2`)

**Date:** 2026-06-09
**Commits reviewed:** ssrl_xrd_tools `b111e5b` (0.40.0), xdart `fc080b4` (0.39.0); working trees clean except CLAUDE.md edits.
**Method:** three parallel deep-review passes (ssrl internals, xdart internals, cross-repo seam), each reading the
actual code, followed by independent spot-verification of every headline claim (the wavelength sentinel chain,
the writer-loop exception window, the legacy alias block, the pyFAI pins, the schema-version read sites were all
re-confirmed by hand). Judged against `roadmap_2026-06-10.md`, both CLAUDE.md files, and
`xrd_two_repo_architecture.svg`. Findings are marked **[V]** verified-in-code or **[S]** suspected.

---

## Executive summary

**Architecture V2 is real, not aspirational.** Every load-bearing claim in the roadmap checks out in code:
streaming is the default executor, `finish()` is fail-loud, the N1 `@source_base` writer+reader round-trip
exists with a thorough test matrix, Pause/Resume and UI-1 are merged, the display core is genuinely pure
(enforced by a subprocess test), dependency direction is one-way and mechanically guarded by AST tests, and the
QtNexusSink signatures match the ssrl protocol. The suites are strong where it matters most: writer invariants,
streaming concurrency (scrambled completion order, measured in-flight bound), persist-before-evict, the
GI-mode-parametrized live≡batch≡reload equivalence spine.

**Two findings should join the Phase D release gate** (both small fixes, both user-visible-wrong-data class):

1. **Wavelength sentinel leaks into display on reloaded scans** (xdart). `LiveScan.mg_args` defaults to
   `{'wavelength': 1e-10}` (= 1.0 Å) and nothing ever updates it [V: only write site is the constructor,
   `scan.py:76-77`]. The *writer* knows this and refuses to persist the sentinel (`nexus_writer.py:1366-1384`),
   but `display_data._get_wavelength` trusts `mg_args` as fallback #2, and its fallback #3 reads
   `entry/calibration/wavelength` — a path the v2 writer never writes. Reloaded frames carry no integrator, so
   on a reloaded v2 scan the 2D Q↔2θ unit toggle converts with λ=1.0 Å → **wrong 2θ axis**, silently. The same
   sentinel feeds `sources.py:148` / `reduction.py:109` energy derivation (future RSM input).
   *Fix:* `_load_from_nexus_v2` populates `mg_args['wavelength']` from `instrument/source/wavelength_A`; give
   `_get_wavelength` the same sentinel-rejection guard the writer has.
2. **`@source_base` stamp failure is swallowed at DEBUG** (xdart `nexus_writer.py:993-996`). If that attr write
   fails, the per-frame relative `source/path` values just written have no base → N1 portability silently broken
   for that file. One-line severity bump (error or raise) — it sits inside a save whose other failures are loud.

Everything else is next-cycle material. The dominant themes across all three reviews:

- **The seam works, but schema ownership leaks across it.** ssrl CLAUDE.md's "io/nexus.py is the single source
  of truth for the on-disk schema" is true for the stacked groups but false for the per-frame source-ref layout,
  which is written only in xdart and read+tested only in ssrl (with hand-built fixtures). One repo can drift
  without the other's tests noticing.
- **The headless write path is incomplete.** ssrl's `NexusSink` writes metadata + stacks + scan_data only;
  source refs, thumbnails, per-frame geometry, and provenance are orchestrated by xdart's writer. A purely
  headless run produces a file on which `get_raw_frame`, FrameView geometry, and N1 portability don't work —
  live≡batch≡reload currently holds only for GUI-written files.
- **Redundant parallel representations.** Two `Scan` classes, two LiveScan→headless adapters (one is dead at
  runtime), two GI-dispatch implementations (one exists only as the equivalence tests' reference), two
  `two_d_kind_from_units` classifiers (already semantically diverged), and the dual display data path
  (publication store + data_1d/data_2d mirrors with three different eviction regimes).

---

## Verdict against the five north-star goals

| Goal | Verdict |
|---|---|
| 1. Headless-first ssrl | **Mostly met.** Zero Qt imports, mechanically guarded. Gap: headless `NexusSink` writes an incomplete v2 record (see S3). |
| 2. Thin xdart | **Largely met.** Writer delegates layout to ssrl; integration has one engine in production. Residues: `LiveFrame.integrate_*` (~280 LOC parallel GI dispatch), `_drop_integrated_rows`, `_quantize_thumbnail`, plan-cache helpers. |
| 3. Robustness | **Strong.** Fail-loud chain is real end-to-end (BLOCKER-2 fix verified). Exceptions: the two release-gate items above, silent monitor-normalization skip, two debug-level data-loss paths in ssrl io. |
| 4. Performance | **Good with two caveats.** Streaming default, flat appends, persist-before-evict engineered properly. Caveats: `ReductionSession._products` retains every 2D result unboundedly; suspected O(N²) pandas `.loc` enlargement on the GUI thread. |
| 5. Expandability | **Met.** FrameSource registry, ReductionSink protocol, PanelKey/layout seams all in place without speculative code. Soft spots: protocol is convention-only (no `runtime_checkable`, optional hooks live in `getattr` strings), and `open_source()` / `LiveScanFrameSource` are not actually used by the GUI runtime. |

---

## What the codebase does notably well

1. **Mechanically enforced architecture** — `test_architecture_guards.py` (AST walk for xdart imports, no-Qt
   import test) and the xdart subprocess purity test for `display_logic.py`. Guardrails as CI properties, not
   conventions.
2. **Fail-loud write discipline** — preflight-before-mutation validators, axis-change full-rewrite refusal,
   label-based append, `@source_base` append conflict rejection, atomic temp+`os.replace`, and the
   surfaced-streaming-failure chain ending in `command='stop'`.
3. **Streaming-engine test design** — chunked path used as a byte-identity oracle; in-flight peak *measured*
   under deliberately scrambled completion order; equivalence spine parametrized over every GI mode combo.
4. **Persist-before-evict as a designed invariant** — enforced at two independent sites with the same `cap-8`
   margin, plus a reload-time forensic diagnostic, plus dedicated tests.
5. **Comment discipline** — review-ID-tagged comments (P2#4, BLOCKER 1/2, O6, L1…) recording *why* and the bug
   history at the exact site, including honest "KNOWN FAIL-OPEN GAPS". Only 7 TODO/FIXME in xdart, all deliberate.

---

## Findings — ssrl_xrd_tools

### S1. Writer-thread kill window → submit deadlock — **major [V]**
`reduction/core.py` `_writer_loop`: the post-write `_emit(progress_cb, …)` and `_clear_source_frame_image(…)`
run in the `else:` clause of the write try/except — exceptions there are **not** caught and escape the loop,
killing the writer thread. The `finally` releases that frame's semaphore slot, but nothing drains further;
after `inflight_max` more submits, `submit()` blocks forever on `semaphore.acquire()` (no timeout, no failure
recorded). xdart's progress callback is signal-emit-only so the window is narrow in practice, but any
user-supplied callback in a notebook can hang a session. *Fix:* wrap the `else:` body in try/except that records
to `self._failure`; optionally make `submit()`'s acquire poll cancel/writer-aliveness.

### S2. Unbounded `_products` retention — **major [V]**
`ReductionSession._products` keeps every `FrameReduction` (including full 2D arrays) for the session's life;
`clear_frame_images` bounds raw images only. A 10k-frame scan with 1000×360 float32 2D results ≈ 14 GB — the
"bounded memory" claim holds for in-flight raw frames, not products. *Fix:* `retain_products: bool | int` knob
(default capped, or off when a persistent sink is attached); document `result.frames` as possibly partial.

### S3. Headless `NexusSink` writes an incomplete v2 record — **major [V]**
Per-frame source pointers, thumbnails, `per_frame_geometry`, positioners-beyond-scan_data, and provenance are
orchestrated only by xdart's `nexus_writer`. Move the "complete v2 frame record" orchestration into
ssrl (shared writer helper / richer `NexusSink`), leaving xdart's sink Qt-signal-only. This simultaneously fixes
the thinness residue and makes ssrl's round-trip tests cover the real on-disk contract (see X1).

### S4. ~275 lines of dead legacy contracts — **major [V]**
`reduction/core.py:136–411`: full legacy `Frame`/`MaskSpec`/`FrameSource`/`Scan` definitions immediately
shadowed by `Frame = ScanFrame` etc. at 405–411. Live-looking dead code; two `Scan.to_metadata` bodies findable
by grep. Delete the block; keep the aliases.

### S5. Two public `Scan` classes — **major [V]**
`io.read.Scan` (read-side handle) vs `core.scan.Scan` (input container) — both are FrameSources, both public.
Rename the read-side one (`ProcessedScan` / `ScanHandle`) before the API freezes.

### S6. `swmr=True` write path likely broken — **major [S]**
`open_nexus_writer(swmr=True)` sets `f.swmr_mode = True` before the resizable datasets exist; HDF5 forbids
creating groups/datasets in SWMR-write mode, so the first frame append should raise. No test exercises it; the
docstring advertises it. Pre-create datasets before enabling SWMR, or remove the flag until implemented.

### S7. Atomic-mode finish failure destroys the whole run — **major-on-paper, rare [V]**
`NexusSink.finish` → on failure calls `abort` → which **unlinks the tmp file**. In atomic mode every
written frame lives in that tmp file, so a failure in the last step (scan_data upsert / rename) deletes all of
it. Keep the tmp as `<name>.partial` on finish-time failure and say so in the raised error.

### S8. Silent monitor-normalization skip — **minor [V]**
`_normalization_for` returns `None` (frame written un-normalized) when the configured monitor value is
missing/zero/non-finite — no log. An explicit exception to the no-silent-wrong-data rule. Warn at least once
per scan, or add a strictness flag.

### S9. Smaller items — **minor/nit [V]**
- Energy sentinel inconsistency (#78 confirmed): `io.read.Scan.energy_keV` typed `float | None` but yields NaN;
  `energy_eV` returns `NaN*1000`; `core.Scan.energy` uses None. Pick None, convert at the read boundary.
- `write_per_frame_geometry`: deletes the existing group, then a `derive_per_frame` failure is logged at DEBUG
  and swallowed — rewrite silently loses previously good geometry. Warn; consider delete-after-derive.
- `_read_scan_data_group`: partial scan_data coverage → all columns silently dropped (no warning).
- `get_raw_frame(allow_thumbnail=True)` default can silently return a dequantized 8-bit thumbnail to notebook
  analyses (DEBUG log only). Warn on fallback, or flip the bare-function default.
- `write_positioners` hardcodes `units="deg"` for every positioner (wrong for translations/temperature).
- N1 cross-OS edges [S]: stored absolute POSIX path opened on Windows is treated as relative (basename fallback
  may rescue); absolute Windows path on POSIX never resolves. A debug log of tried candidates would help support.
- pyFAI pin: ssrl allows `<2026` (re-admits 2025.12) while xdart pins `<2025.12`. The known breakage is
  pyFAI-calib2/silx-GUI (xdart-side), so ssrl's pin is probably fine — but verify nothing ssrl-side broke on
  2025.12 before releasing, and consider an explanatory comment mirroring xdart's.
- `viz/__init__` eagerly imports matplotlib+plotly → both are hard base deps of the "headless core". Lazy
  module `__getattr__` would slim installs.
- Per-frame file reopens: `io.read.Scan.iter_chunks` / `ProcessedNexusSource.to_scan` reopen the `.nxs` (and
  master) per frame — O(2N) opens; group consecutive frames per resolved master.
- Module splits (already roadmapped): `io/nexus.py` (2562 LOC, four concerns) and `reduction/core.py` (2345 LOC).
- RSM tests remain shape-level (no physics oracle; xrayutilities paths skip silently when absent) — keep tied
  to RSM feature work as planned.

---

## Findings — xdart

### X-gate-1. Wavelength sentinel chain — **major [V]** *(release gate; see executive summary)*

### X-gate-2. `@source_base` stamp failure swallowed — **minor severity, major consequence [V]** *(release gate)*

### X1. Dual display data path — **major [V]** *(known, in-progress; this confirms the concrete mechanism)*
`_data_snapshot` computes render-eligibility as the **union** of `data_1d`/`data_2d` keys and publication-store
availability, but PLOT_1D and RAW_2D draw from the mirrors while CAKE_2D draws from publications — under three
different eviction regimes (data_1d ∞, data_2d FIFO-40 + hydrated-raw LRU-8, publication heavy-payload bound 64).
A frame can be render-eligible via publications while its mirror entry was evicted. The `_two_d_axes_match`
guard prevents the worst (silent axis blending) and all observed write sites update both stores under one
`data_lock` — but consistency is maintained by convention at every site, not by construction. Finishing the
"publications as sole display contract" migration is the right fix and is already the stated direction.

### X2. Parallel GI-dispatch implementation kept as test reference — **major-arch [V]**
`LiveFrame.integrate_1d/2d` (~280 LOC: GI mode switch, kwarg filtering, fiber-integrator construction, monitor
normalization) is never called in production (every `add_frame` passes `calculate=False` — verified at all four
call sites) but is the "live serial" *reference* in the equivalence spine (`test_gi_batch_real_data.py:872`).
The equivalence guarantee is thus test-enforced across two implementations rather than shared-code-enforced.
Either route it through the ssrl plan path or demote it explicitly to a test fixture; add a
`_process_one`-vs-sink equivalence test either way.

### X3. GI fail-open gaps — **minor [V]** (documented, unfixed)
Multi-master Eiger with varying incidence, and Image-Directory sweeps, still freeze the GI grid from chunk 1
(`image_wrangler_thread.py:1160-1168`). Honestly documented in-place; keep on the roadmap so the comment isn't
the fix.

### X4. Suspected O(N²) scan_data accumulation on the GUI thread — **minor [S]**
`update_data` does `sd.loc[idx] = ser` per frame (pandas enlargement). Matches the deferred "if 1D still slopes,
suspect GUI-side accumulation" note. Profile during Phase D B3/B4; fix is batched appends or a list-of-rows
buffer flushed periodically.

### X5. Smaller items — **minor/nit [V]**
- Threading: clean overall — no cross-thread `QTimer.start()` anywhere; the 0.37.1 bug class is structurally
  prevented; single-writer invariant holds; pause/resume ordering correct with every examined race already
  closed and documented. `_wait_if_paused` busy-polls at 50 ms (an `Event.wait()` would be cleaner).
- `QtNexusSink._registry` can pin LiveFrames for frames that fail/cancel mid-flight; clear in `abort()`/`finish()`.
- `_absorb_chunk`'s coalescer is a debounce while `update_data`'s is a throttle — opposite semantics in adjacent
  files invites regressing the "throttle not debounce" lesson; align or comment.
- Stale comment says `data_1d`/`data_2d` are "both bounded" (`static_scan_widget.py:680-688`) — contradicts the
  deliberate unbounded-data_1d decision; fix the comment.
- `display_constants → integrator` layering inversion (pulls pyFAI into a constants module; the reason
  display_logic inlines its glyphs). Make constants leaf-level.
- Thinness residues to move to ssrl: `_drop_integrated_rows` (see X-seam below), `_quantize_thumbnail`,
  axis-signature helpers, `StandardPlanCache`/mask/GI-default helpers in `modules/reduction.py`.
- 3 stray `print()`s in `gui/widgets/`; `ipykernel` in runtime deps; audit whether `xrayutilities`, `silx`,
  `joblib`, `lmfit`, `matplotlib` are still direct xdart deps.
- God modules (already RESTRUCTURE-TODO'd): `image_wrangler_thread.py` (2606), `h5viewer.py` (2539).
- Test gaps: wrangler run loops end-to-end, h5viewer `_absorb_chunk` gate directly, `NexusViewerController`,
  pause against a *real* streaming session (current 11 tests use stubs).

---

## Findings — the cross-repo seam

### C1. Schema-version forward-compat: written, never read — **high [V]**
`ssrl_schema_version = 2` is stamped by ssrl's writer, and **no reader in either repo ever reads it**. A v3 file
hits today's readers with no warning — failures will be downstream KeyErrors or silently missing features.
*Fix:* `read_scan`/`open_scan`/`read_frame_view` warn (or raise behind a flag) when file version > supported.
Cheap, and the whole point of having stamped it.

### C2. Per-frame source-ref schema split across repos — **high [V]**
`frames/frame_NNNN/source/{path,frame_index}` + `entry/@source_base` are *written* only by xdart
(`_write_source_ref`/`_write_per_frame_metadata`), *read* and *tested* only in ssrl — whose N1 tests hand-build
the layout in raw h5py. A key rename in xdart keeps ssrl's tests green while real files break. *Fix:* add
`write_frame_source_ref(...)` (+ the `@source_base` stamp/conflict check) to `ssrl io/nexus.py`; xdart calls it;
ssrl tests use it. Combines naturally with S3.

### C3. `_drop_integrated_rows` hardcodes ssrl's row-aligned dataset names — **medium-high [V]**
xdart's replace path rebuilds ssrl-owned `integrated_*` groups with a hardcoded
`{"frame_index","intensity","sigma"}` row-aligned set. If ssrl adds another N-aligned dataset, xdart copies it
unfiltered → silent length mismatch. Move row-dropping into ssrl io.nexus, which owns "what is row-aligned".

### C4. No runtime API-compat check; pip floor is bypassed by the dev workflow — **medium [V]**
The `ssrl_xrd_tools>=0.40.0` floor is currently correct, but with editable installs from sibling clones (the
documented workflow) pip floors protect nothing: a stale ssrl checkout + new xdart crashes at first write.
*Fix (cheap):* startup assertion in `xdart_main.py` against a single `MIN_SSRL_VERSION` constant + a tiny test
asserting that constant equals the pyproject floor; optionally `hasattr` capability probes for the newest
load-bearing symbols.

### C5. Dead seam: `LiveScanFrameSource` / `open_source()` unused at runtime — **medium [V]**
Two complete LiveScan→headless adapters exist (`modules/sources.py`, `modules/reduction.py`) with duplicated
`_scan_data_row`/path/wavelength logic and subtle differences. The runtime uses only the `reduction.py` path;
`LiveScanFrameSource` is exercised solely by its own test, and xdart never calls ssrl's `open_source()`
registry. Either build the runtime `Scan` via `LiveScanFrameSource.to_scan()` (one adapter) or delete it and
correct ARCHITECTURE_V2.md, which currently documents a seam the code doesn't use.

### C6. `two_d_kind_from_units` duplicated and diverged — **medium [V]**
xdart `display_logic.py:500` (string results, substring matching) vs ssrl `core/frame_view.py:262` (enum,
`startswith`, distinguishes `QTOT_CHIGI`). Same on-disk strings, two classifiers, already semantically
divergent on `qtot_*` and `exit*` units. Re-point display_logic at the ssrl function (ssrl `core.frame_view` is
Qt-free; relax the purity guard to allow `ssrl_xrd_tools.core` or map enum→strings at the edge).

### C7. No CI in either repo — **medium [V]**
The cross-repo contract is protected only by manually running both suites. Even a minimal GitHub Actions matrix
(ssrl full; xdart `-m display_logic` + sink/adapter subset against ssrl head) would catch the entire class of
drift this review keeps flagging. Add a `tests/test_sink_contract.py` in ssrl driving a streaming session
against a stub six-method sink (begin/write/replace/finish/abort/worker_process) to freeze the duck contract —
including which thread each hook runs on, which today lives only in QtNexusSink's docstring.

### C8. Smaller seam items — **low [V]**
- Private reach: `image_wrangler.py:26` imports `_extract_scan_info` (excluded from ssrl's `__all__`); promote it
  (or a `predict_spec_sidecar()` wrapper) to public. `relative_source_path` is public in `io/__init__` but
  missing from `read.py.__all__`; xdart imports it from the submodule — align the publicness signal.
- `ReductionSink` protocol is not `runtime_checkable` and its three optional hooks exist only as `getattr`
  strings; document them in the protocol and add the checkability.
- Trivial duplicates: `_readonly_mapping`, `_is_eiger_master` (byte-equivalent in both repos); `run_stitch`
  name collision (different things in each repo).
- Doc staleness: xdart `ARCHITECTURE_V2.md` still says "spike — do not merge"; ssrl `PUBLISHING.md` cites
  `>=0.2.0` floors and an `xdart-0.15.0` wheel; ssrl CLAUDE.md's "single source of truth for the schema" is
  overstated until C2 lands. Sweep at release.

---

## Consolidated priorities

**Phase D release gate (do before `architecture-v2 → dev`):**

| # | Item | Where |
|---|---|---|
| 1 | Wavelength sentinel: repopulate `mg_args` on v2 reload + sentinel-reject in `_get_wavelength` | xdart `scan.py` / `display_data.py` |
| 2 | `@source_base` stamp failure: DEBUG → error/raise | xdart `nexus_writer.py:993-996` |
| 3 | Stale-comment + doc sweep (ARCHITECTURE "do not merge", PUBLISHING floors, bounded-cache comment) | both |
| 4 | Decide on the pyFAI `<2026` vs `<2025.12` pin inconsistency | ssrl `pyproject.toml` |

**Next cycle, high value (rough order):**

1. Complete-v2-record orchestration into ssrl (`NexusSink` parity) + `write_frame_source_ref` + row-drop helper
   move — fixes S3, C2, C3, and most of the thinness residue in one arc.
2. Reader-side schema-version check (C1) — trivial, do alongside #1.
3. Streaming hardening: writer-loop exception wrap (S1), `_products` retention knob (S2), atomic-mode partial
   preservation (S7), sink-finish result consistency.
4. Startup min-version assertion + floor-sync test (C4); sink-contract test in ssrl (C7); CI skeleton.
5. Finish publications-as-sole-display-contract (X1); retire or productionize `LiveFrame.integrate_*` (X2) with
   a `_process_one`-vs-sink equivalence test.
6. Unify the LiveScan adapter (C5); re-point `two_d_kind_from_units` (C6); delete the legacy block (S4); rename
   `io.read.Scan` (S5).
7. SWMR: fix or remove (S6). Monitor-normalization warning (S8). Energy sentinel (#78/S9).
8. Profile scan_data accumulation during B3/B4 (X4); the already-roadmapped module splits.

A separate companion document, `greenfield_design_2026-06-09.md`, addresses what a from-scratch design would do
differently and how those choices map onto incremental moves for this codebase.
