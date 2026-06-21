# Streaming reintegrate — the proper D1 fix (shadow-dataset + atomic swap)

**Date:** 2026-06-20 · **Status:** designed, ready to implement (its own focused effort).
**Goal:** make Re-Integrate stream to disk incrementally so memory is bounded for
ANY scan size — closing the D1 gap by extending the live-streaming architecture
to the reintegrate path. Vivek: "fix it properly, don't keep coming back."

## Why D1 exists (and live doesn't have it)
Live acquisition streams each frame through `QtNexusSink` → persisted → the 64-cap
`persist-before-evict` eviction keeps memory flat. **Reintegrate is the one path
that bypasses streaming:** it accumulates every recomputed frame and does ONE
end-of-run `save_to_nexus(replace_frame_indices=…)`, because a *shape-changing*
reintegrate (changed npt/range — the common retune) must write all frames
together (`_select_frames_to_write` rejects mixing fresh + stale rows). So nothing
is persisted until the end → `persist-before-evict` can't evict → all N pile up.

The Jun-20 mitigations (committed) bound it for the user's 651-frame scans (drop
map_raw + the stale 2D slab; post-save eviction) but 2D-reintegrate results still
scale with N. This design removes the scaling.

## Approach: shadow group + atomic swap (the user's shadow-dataset idea)
1. **Start:** create an empty shadow stack group, e.g. `/entry/integrated_1d__reint`
   (and `/entry/integrated_2d__reint` for a 2D pass).
2. **Per batch:** APPEND the batch's recomputed rows to the shadow group (the
   existing append/streaming write path, targeting the shadow `group_path` — rows
   are uniform within one pass, so append is legal; no shape-mixing because the
   shadow starts empty). Then `mark_persisted` those frames + `evict_persisted_beyond_cap`
   → memory stays at ~cap, exactly like live.
3. **End (success):** atomic swap, under the H5FilePool pause:
   `del /entry/integrated_<dim>`; `h5f.move("…__reint", "/entry/integrated_<dim>")`.
   Then write the reduction provenance (bai_*_args) for the new params. The
   untouched dimension (e.g. 2D on a 1D-only pass) is never moved.
4. **Stop / error:** delete the shadow group; the original `/entry/integrated_<dim>`
   is untouched → "restore on abort". A stopped reintegrate leaves the prior
   result intact (vs today's best-effort partial — cleaner, and the only legal
   option for a shape change anyway).

## Writer changes
- `write_integrated_stack` / `save_scan_to_nexus`: accept a `group_path`/suffix
  override so the reintegrate can target `…__reint`, reusing the append cursor
  (`NexusWriteCursor`) for incremental writes.
- New `_swap_integrated_group(h5f, shadow_path, final_path)`: del-final + move-shadow,
  all inside one open handle (HDF5 link ops are in-file atomic); on a crash between
  del and move, the next open sees a missing final + an orphan `…__reint` → a
  startup recovery (adopt the shadow, or warn) — add a guard in the reader.
- Keep the strict validators: the shadow is validated as a fresh append stack; the
  swap only relinks groups (no row-shape relaxation). Do NOT loosen
  `_select_frames_to_write` / `_require_uniform_axes_*`.

## Reintegrate-loop changes (`scan_threads._reintegrate_all`)
- Replace "accumulate all → end replace-save" with: per batch, reduce → publish
  (display) → append to shadow → mark_persisted → evict. End: swap (success) or
  drop-shadow (stop/error).
- The mitigations stay (drop map_raw; for 1D-only the 2D group isn't touched at
  all — no shadow_2d, no rewrite).
- Display is unaffected — it already renders from the in-memory PublicationStore
  (bounded), not from the frames being evicted.

## Verification gates (must pass before merge)
- **Equivalence spine** (`test_gi_batch_real_data::…equivalence`): the swapped
  result must byte-match the current end-replace-save result.
- **Abort/rollback test:** stop mid-reintegrate → original `.nxs` integrated
  stacks unchanged; the shadow group is gone.
- **Memory-bounded test:** reintegrate N≫cap frames → `_in_memory` stays ≤ cap+ε
  throughout (not N).
- **1D-only preserves 2D:** a 1D reintegrate never alters `/entry/integrated_2d`.
- **Crash-between-del-and-move:** a file with an orphan `…__reint` opens sanely.

## Effort / risk
Substantial + writer-guardrail-sensitive. Do it as one focused effort on the
committed checkpoint base; implement incrementally (writer building block + its
tests first, then rewire the loop), gated on the spine at each step.
