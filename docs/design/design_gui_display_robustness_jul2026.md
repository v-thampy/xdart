# GUI display-pipeline robustness — diagnosis, quick wins, and v1.1 structure

**Date:** 2026-07-07 · **Status:** design + handoff. Companion to
`runend_overlay_catchup_spec_jul2026.md` (PERF-3 tail) and the OV rows in
`live_findings_ledger.md`. Sites cite HEAD `e3fa7b9e`.
`SSW` = static_scan_widget.py, `DFW` = display_frame_widget.py, `DP` = display_publication.py,
`DL` = xrd_tools/session/display_logic.py, `DOU` = display_overlay_utils.py.

## 0. Scope and immediate trigger

Immediate trigger: the maintainer's stress test — **Overlay + Share Axis + mid-run 2D display-unit
flip (Q-χ → 2θ-χ) on a 651-frame live run** — produces (a) a Q-labeled waterfall forced to a
±70 range with the data squished near 0, (b) a stale "2θ-χ" combo over a Q-rendered cake, and
(c) blank horizontal bands in the overlay plus the (already-ledgered PERF-3) row-count cap.
This is a REGRESSION by composition (§2). But the deeper ask: the OV/overlay family
(OV-1..7, S-14, S-16, S-17, BL-6, and now this) has produced ~12 bugs in five weeks. §3 explains
why structurally; §4 gives quick wins; §5 the v1.1 redesign.

## 1. The pipeline as-built (map)

Producer side: wrangler thread → `_BoundedFrameHandoff` (cap 128, drop-oldest) →
`update_data` drain (GUI) → `PublicationStore` (byte-budget, Item-1 1 GiB) → render tick.
Render: `DP.plot_payload` builds rows per label (per-row: unit conversion → norm → bkg → slice
projection) → `DL.accumulate_waterfall` merges into `WaterfallHistory` (geometric row buffer,
BW-A6) → draw (per-trace curves ≤15, else waterfall image, paint decimated to
`MAX_WF_ROWS=256`). Side channels: `FrameHydrationWorker` (per-label tiers, RL-1 gates),
`_LoadFramesWorker` (bulk browse chunks), `AggregationWorker` (whole-scan), the H30 generation
scheduler, ≥5 debounce/coalesce timers (selection 100 ms, load 100 ms, update coalescer, live
2 Hz throttle, hydration quiet-timer), and the Share-Axis geometric align
(`_align_plot_under_cake`, DFW:2848 — NOT a pyqtgraph axis link; it transfers the cake's numeric
range onto the 1D plot with x-autorange disabled).

## 2. The unit-flip regression — root cause (diagnosed, 3-commit composition)

No single commit; three interact:

1. **`efa0b396` (Jul 1) — mid-run wavelength blackout = the primary regressor.**
   `_get_wavelength` early-returns None while `_run_writing` (display_data.py:1342-1343; HDF5
   fallback gated off mid-run; the 1e-10 mg_args sentinel is rejected). Hydrated rows have
   `raw_ref=None` (frame_publication.py:254/449/475), so mid-run the per-row Q↔2θ display
   conversion is available for frame-backed rows and unavailable for hydrated rows — **the same
   append batch mixes units**, and `_apply_image_unit_2d` (DP:651-657) silently renders the cake
   in Q under a "2θ-χ" combo (the stale-combo signature). Pre-`efa0b396`, the file fallback
   usually resolved between write bursts — this "worked before".
2. **`552365eb` (BL-6, Jul 4) — value-mismatch reinterp turns mixing into destruction.**
   `accumulate_waterfall` now reinterps same-size rows on a VALUE mismatch (DL:718-722). Correct
   for same-unit range shifts; across units `np.interp(base_x≈1-8.5 Å⁻¹, x≈10-55°, y)` has
   disjoint domains → clamps to a constant → the **blank bands**, permanently appended. The unit
   guard (`history.unit != unit` relabel at DL:693; seed unit-equality at DP:1100-1106) sees only
   the batch-level first-row unit — it cannot catch per-row mixing. The grid key is unit-blind by
   design (`_axis_token_from_text` maps Q and 2θ both to "radial", DOU:158-161) — correct per the
   OV-5/OV-6 contract (unit toggle must RELABEL, not reset), but it means nothing downstream
   rejects a cross-unit row.
3. **`f47790f4`/`5c590779` (Jun 16, pre-existing) — Share-Axis align displays the damage.**
   The align is combo-keyed, applied AFTER the payload is built (DFW:3068), silently force-selects
   `plotUnit` (blockSignals, no `_last_plot_unit` update), and imposes the cake's numeric range
   with autorange OFF and **no unit-agreement check** — hence a Q-labeled panel pinned to a
   2θ-derived ±70 range.

Screenshot 3's ~290-row cap is the separate, already-ledgered PERF-3 handoff/backfill deferral
(the unit-flip stress adds GUI lag → more cap-128 drops); the run-end catch-up spec covers it.

### Fixes (F1-F4) — **CORRECTED 2026-07-08 after adversarial verification; use the deferred-ledger forms, not the text below**

> F1 as literally written is a NO-OP (the consult already exists); the real form is a run-start
> stamp of a widget-scoped `_run_wavelength_m`. F4 as written creates a 0-ms render cascade
> (`singleShot(0)` bypasses `_in_update`); the safe form mirrors `_last_plot_unit` only. F2 is ON
> HOLD (site attribution wrong + an OV-contract trap that would freeze the overlay on a clean unit
> flip). F3(b) is apply-with-care. Corrected forms live in `deferred_ledger.md`. ALSO SUPERSEDED:
> the dominant live stall is NOT paint/cadence — multiple faulthandler dumps pin it as MEM-1 [14]:
> per-render whole-selection publication rebuilds in `display_controllers._data_snapshot` +
> `_store_first_publication_items` (O(N) `resolve_frame_data_for_widget` + `publication_from_frame_view`
> + `validate_publication` per tick under auto-last Overlay). That stall is also why the run-end
> catch-up cannot quiesce and the tail stays short.

- **F1 (root, ~5-10 lines, low risk):** run-scoped wavelength. Cache the run's PONI wavelength at
  run start (widget-level `_run_wavelength_m`, or stamp `scan._persisted_wavelength_m`) and
  consult it in `_get_wavelength` BEFORE the `_run_writing` early-return. Kills mixed batches AND
  the stale-combo cake at the source.
- **F2 (~15 lines, low-med risk):** unit-safe accumulate. In `DP.append_row`, when the row's
  converted axis unit ≠ the batch/accumulator unit, SKIP the row this render (it re-arrives
  converted later — same policy as S-17 empty rows). Belt-and-suspenders: gate the BL-6 interp
  (DL:718-722) on unit equality by threading the incoming unit into the loop guard. Must keep
  `test_bl6_overlay_xgrid.py` green (legit BL-6 case is unit-equal by construction).
- **F3 (~20 lines, med risk — additive guards only, no geometry changes):** Share-Axis (a) skip
  the align + re-enable x-autorange when the cake's RENDERED unit ≠ the 1D's rendered unit (stash
  the rendered axis with the image payload); (b) key `_current_image_axis_key` off the last
  rendered cake axis, falling back to the combo.
- **F4 (~4 lines, low risk):** when `_apply_share_axis_state` silently switches `plotUnit`, also
  set `_last_plot_unit` (mirror DFW:4751) and request one follow-up render (0-ms single-shot,
  existing re-entrancy guard).
- **Offscreen test** (pattern: test_aggregation_wiring + test_bl6_overlay_xgrid): duck widget,
  `_run_writing=True`, Overlay + share-axis + slice; interleave frame-backed (`raw_ref` with
  wavelength) and hydrated (`raw_ref=None`) publications; flip `imageUnit` mid-stream. Assert:
  count monotonic to N; `history.x` strictly monotonic, single unit; every row `np.ptp(row) > 0`
  (no clamped bands); with wavelength withheld, the 1D viewport never takes the foreign range
  (autorange stays on / viewRange within data span).

Contract note: the OV acceptance contract requires the accumulator to SURVIVE a display-unit flip
(relabel, not reset). F1/F2 preserve that; do NOT "fix" this by adding unit to the reset key.

## 3. Why this family keeps breaking — six structural fragilities

- **F-A. Rows are stored post-transform.** The accumulator holds display-space rows (unit-,
  norm-, bkg-, slice-transformed). Every display toggle must then be classified — relabel vs
  reinterp vs reset vs per-row skip — at ≥3 code sites, and any miss = silently mixed rows.
  Today's bug, S-16's mixing, BL-6's misgrid, and M5's double-bkg are ALL this fragility.
- **F-B. Identity is inferred from live widget state at render time.** Combo text, `plotUnit`
  index, `overlay_axis_kind` UI reads (the earlier N5 note), the share-axis combo key. State can
  be stale relative to the rendered data (this bug), or mutated mid-render.
- **F-C. Reset/append decisions are scattered.** At least six sites decide accumulator lifecycle:
  grid reset-key compat (DL), seed gating + in-batch clear (DP), norm-change and method-switch
  wipes (DFW:3806, DP:750-759), S-14 seen-set (SSW rescope), reintegrate-finish, Clear. Each was
  added by a different fix; the acceptance contract lives in a ledger doc, not in code. The
  round-4 finding that two history-only wipes forget the pins is this fragility.
- **F-D. Run/selection choreography is an implicit timer web.** Debounced one-shots that rewrite
  selection state race feature-level operations (the Item-2 autopsy: two 100 ms echoes killed the
  deferred render). Nothing owns "the run has ended and the display is quiet".
- **F-E. Display transforms depend on ambient lookups.** Wavelength (this bug), norm channel
  (S-16's combo reset), monitor values — fetched at render time from mutable context instead of
  carried with the data they describe.
- **F-F. Mechanism-not-outcome tests.** Item-2's "a render fires", the pre-ace883a P1 test that
  modeled store-only availability, the BL-6 lane test that missed the adapter half — the family's
  regressions repeatedly pass tests that assert the mechanism moved, not that the picture is
  right.

## 4. Quick wins (v1.0.x — small, independently landable)

1. **F1-F4 above** (the regression fixes) — combined ≈ 1 day incl. the test.
2. **Run-end catch-up** per `runend_overlay_catchup_spec_jul2026.md` (S, spec'd, verified sound
   with the 3-tuple row-id correction).
3. **Codify the acceptance contract as an executable invariant harness.** One offscreen harness
   driving scripted event sequences (publish, evict, click, deselect, unit toggle, norm change,
   rescope, reintegrate, hydration completion) against the real adapter+accumulator, asserting
   after EVERY step: count monotonic except at explicitly-allowed reset causes; single monotonic
   x grid; no constant-clamped rows; pins ⊆ history. Each ledgered OV bug becomes one ~10-line
   sequence. This converts the contract from prose to CI and is the highest-leverage test
   investment available (S/M, ~1 day).
4. **Debug-build tripwires** (S): assert-on-mix in `accumulate_waterfall` (row unit ≠ history
   unit and no relabel intent → log ERROR + skip, never interp); assert the GUI thread never
   blocking-acquires `file_lock` while `_processing_active` (would have caught BB-1 years early);
   both behind `XDART_DEBUG_DISPLAY=1`.
5. **Heartbeat + render metrics** (landed as BB-1c): add `render_ms` + accumulator row count to
   the XDART_PERF summary so the 125→180 s overlay overhead can be attributed (the paint is
   decimation-capped at 256 rows, so the naive O(N)-paint theory is wrong; suspects are
   accumulator bookkeeping, per-row conversion, and GIL contention with the reduction worker
   THREADS — measure before optimizing; adaptive cadence remains the likely lever).

## 5. v1.1 structural redesign (one branch, sequenced)

**V1 — Canonical-grid accumulator (the deep fix for F-A/F-E).** Store rows in the
acquisition-native grid (the 1D integration grid; slice rows in the 2D radial grid) with per-row
carried metadata: source unit, wavelength, norm channel + value applied, bkg recipe, slice
projection id. ALL display transforms (unit relabel, norm, bkg, log/sqrt) become pure functions
applied at draw time from carried metadata. Unit toggle = view change (relabel the axis, no data
touch); norm change = re-render, no reset (S-16 and its residuals dissolve); mixed batches become
impossible (rows are never stored converted). The BL-6 reinterp survives only for genuine
same-unit grid drift. Touches DL + DP payload build + DFW draw; the WaterfallHistory schema gains
a metadata column it already half-has (round-11 added per-row metadata). Estimate M/L (2-4 days)
— by far the best payback-per-risk of the structural items.

**V2 — Single AccumulatorLifecycle owner (F-C).** One object owning reset/append/relabel with an
explicit cause enum (`CLEAR`, `INCOMPATIBLE_GRID`, `REINTEGRATE`, `NORM_CHANGE`,
`SAME_NAME_RERUN`); every current wipe site calls it with a cause; pins and history always reset
TOGETHER through it (closes the round-4 pin-survival hole by construction); the invariant harness
(§4.3) enumerates causes exhaustively. S/M once V1 lands.

**V3 — Carry identity with payloads (F-B).** The image payload records its rendered axis
unit/kind; the 1D payload likewise; Share-Axis and every "what unit is on screen" consumer read
the payload, never a combo. F3 is the tactical version; V3 makes it the rule.

**V4 — Run-boundary sequencer (F-D).** Replace the debounce-echo web at run start/end with an
explicit little state machine (RUNNING → FLUSHED → RECONCILED → QUIET) emitting one signal per
transition; the run-end catch-up, the Scans-panel follow, auto-last collapse, and any future
run-end feature subscribe to QUIET instead of racing timers. S/M; de-risks every future run-end
feature.

**V5 — Per-file lock registry** (already on the v1.1 board from BB-1): display reads of file A
never contend with the writer of file B; also enables chunked aggregate lock holds (S-10
residual). L, own design checkpoint — the concurrency-spine change.

**V6 — Display fuzzer (F-F insurance).** Randomized event-order generator over the §4.3 harness
vocabulary, run N seeds in CI nightly; any invariant violation shrinks to a replayable sequence.
The OV family is precisely the class of bug this finds before users do. S/M after §4.3.

Sequencing: §4 items pre-tag / v1.0.x → V1 → V2+V3 (ride V1's schema) → V4 → V6 → V5 (own
checkpoint). V1-V4+V6 ≈ 1-2 weeks of focused work; V5 separate.

## 6. Verification snapshot folded in (2026-07-07, at `e3fa7b9e`)

Closed and verified this pass: **BB-1** (`40040aa8` — display snapshot skips the frame walk;
non-vacuous test; writer/headless callers keep full inputs; maintainer live-verified);
**q-wedge blocker** (`4ee01755` — input shift via `_integration_azimuth_range` in the q/2θ
branch, real-pyFAI 1D≡2D test, G18 script gained the q-wedge case); **NeXus handoff + close
paths + bg browse-cancel** (`8f6a769e` — `_active_scan` set in `_initialize_scan` with safe
gating, both close-timeout paths detach+retain, `set_bg_file`/`set_bg_dir` guarded, BW-A4 row
honestly amended); **MIGRATION item 18** (reworded honestly); **publish safety** (tracked RC-8
recipe marked SUPERSEDED; strip covers docs/design + docs/history; old-objects question
documented as an explicit maintainer decision); **ledger truth** (MS-1 reopen, three SHAs, G11
dedup, G18 validator script + smoke test). New minors from this pass: `e3fa7b9e` iterates
`frames._in_memory.values()` without `_cache_lock` in `_raw_files_from_scan` (streaming-flush
race → RuntimeError; snapshot under the lock); G18 has no out-of-domain wedge case though
MIGRATION says it "pins the observed behavior"; no G-item exercises live NeXus Overall;
`d032c6a6` prefix-fuzzy scan match can highlight the wrong scan (cosmetic). Remaining before
tag: the unit-flip fixes (§2), Session-1 (incl. G18 real-data run), re-freeze + full battery at
the final SHA, and the maintainer's publish decision.
