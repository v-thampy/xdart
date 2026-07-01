# Review 2026-06-28 — stitch / RSM / corrections (headless P1–P6 + Codex fixes)

**Scope:** independent deep review at HEAD `8addcdd` of the headless stitching + RSM +
corrections work (`stitching_rsm_build_plan.md` P1–P6) and the fixes Codex/prior sessions
landed on top (commit `d090aba` "Fix 4 review findings"; `8addcdd` accumulator unification +
footprint flip; the uncommitted→committed `run_stitch` generator refactor). Four areas reviewed
by isolated agents that ran code (pyFAI 2025.1.0 + xrayutilities 1.7.12), not by reading alone.

**Verdict:** the implementation is in **strong** shape. The most serious issue — a
stitch↔RSM merge-accumulator inconsistency — was already found by a prior session, pinned as a
strict-xfail, and **fixed in `8addcdd`**; I re-verified the fix is correct and complete. The
base corrections are bit-exact against pyFAI, persistence round-trips, and the full core suite
is green (**1347 pass / 3 skip** here = the doc's 1348✓; skips are env/real-data-gated). **One
real `[major]` remains open** (a NaN-collapse guard gap on the legacy stitch path), plus a
couple of minors/nits. The GI *absolute* conventions remain correctly deferred to a real-data
GIXSGUI gate.

---

## 1. Open findings (action needed)

### [MAJOR] Legacy (non-`Diffractometer`) stitch path still silently collapses NaN rotations
`src/xrd_tools/integrate/multi.py::create_multigeometry_integrators` (sets
`ai.rot1 = base_rot1 + deg2rad(r1_deg)` with **no finite check**) and the consumer
`src/xrd_tools/analysis/plans.py` `run_stitch` legacy `else` branch (~l.432:
`rot1 = _metadata_series(src, labels, plan.rot1_key)` → `create_multigeometry_integrators`,
no `isfinite` guard).

`d090aba` correctly closed this NaN-collapse class on the **two** other vectors — the
`Diffractometer` path (`multi.py:151` finite check in `create_multigeometry_integrators_from_geometry`)
and the monitor series (`plans.py:297`). It did **not** cover the legacy `rot1_key`/`rot2_key`
fallback. That path is reachable: a `CompositeFrameSource` over a heterogeneous group NaN-pads a
member missing the rotation motor (the intended contract), the motor key is then
*present-but-NaN*, slips past the missing-key guard, and `_metadata_series`'s motors fast-path
returns e.g. `[10.0, nan]`. pyFAI then collapses the NaN-rot frame (`qmax≈0`, frame dropped from
the merge) **with no error** — the exact symptom `d090aba` rates `[major]` for the twin path.

*Likelihood:* the app default is now the psic `Diffractometer`, so this is the *fallback*
(`base_poni` + `rot1_key`, `diffractometer=None`) crossed with the *new* multi-source grouping —
plausible, not the hot path. Silent wrong output, so still `[major]`.

*Fix:* add the same fail-loud finite check on `rot1`/`rot2` in `create_multigeometry_integrators`
(covers every caller of the legacy primitive), mirroring `multi.py:151`. Add a regression test on
the legacy path (the existing `test_stitch_geometry.py:71` `match="non-finite"` covers only the
`_from_geometry` path).

### [MINOR] `discover_scans` silently degrades to lexicographic order if `natsort` import fails
`src/xrd_tools/sources/discover.py` `_walk_files` wraps `from natsort import os_sorted` in
`try/except Exception` and falls back to plain `sorted()`. `natsort` is a hard core dependency,
but when it's genuinely missing/broken, discovery returns `scan_1, scan_10, scan_2` instead of
natural order — a **mis-ordered stitch/RSM group with no warning** (this is exactly how the test
failed in a clean env). `except Exception` also swallows unrelated errors.

*Fix:* import `natsort` at module top (let an `ImportError` surface, since it's required), or at
minimum `logger.warning` once in the fallback so a mis-ordered group is diagnosable.

### [NIT]s (optional)
- `src/xrd_tools/io/nexus_record.py` `_partition_selected_local_labels` (~l.419): the ambiguous
  case (a bare list — not a composite — *with* `selected_labels`) returns `None` → records ALL
  frames. No current caller hits it (`run_stitch` always wraps groups in `CompositeFrameSource`
  before harvest), but a future caller would silently widen to every frame. Consider raising.
- `src/xrd_tools/integrate/stitch_hist.py` `xu_q_frames` still materializes internally and keeps
  its own pre-refactor monitor-validation block rather than sharing `_normalization_array` /
  `_iter_image_integrator_pairs`. It's the **dead** `xu_hist` provider (`backend` raises), so
  harmless, but it now diverges from the streaming pattern and duplicates the monitor guard.
  Unify when `xu_hist` is wired (P3c).
- RSM empty bins changed `0 → NaN` (Σraw/Σnorm). Behaviour-improving and documented
  (`gridding.py:46-55`); `combine_grids` already `nan_to_num`s. Any downstream consumer that
  treated RSM empty bins as `0` should be aware.

---

## 2. Verified CORRECT (ran code, not just read)

**Accumulator fix (`8addcdd`) — correct & complete.** `rsm/gridding.py` `_feed_pair`/
`_pair_intensity` now accumulate `Σraw` (bare image) and `Σnorm` separately, volume = `num/den`.
Fed `raw=true·C, norm=C` to the *real* gridder → recovers `true` exactly (old `Σ(raw·w)/Σw`
would return `true·ΣC²/ΣC`). The de-xfail'd `test_merge_accumulator_consistency.py` passes for
**both** pipelines; `weight=1` still reproduces the count-mean; bad pixels drop from **both**
channels; `grid_img_data` (non-streaming) and `combine_grids` are consistent; **full-tree grep:
zero residual `Σ(raw·w)`**. The `gridding.py:51` comment was corrected.

**GI footprint flip + grazing physics — correct.** `footprint_weight = 1/sin αi` (boost; so
corrected = raw·sin αi), **consistent across all three sites** (primitive `footprint_weight`,
`gi_normalization`, the RSM `gi_grid_weight` path) and the stitch path — all `1/sin αi` through
the single `gi_normalization`. xu cross-check (Si@10keV): δ/β/αc/μ match; Fresnel `|T|²` peaks at
αc (Yoneda); refraction is a **position** correction (`refract_q` shifts qz, →0 above αc, NOT a
weight); absorption sign correct. The `sample_orientation`/`tilt` *absolute* conventions are
correctly flagged in-code as real-data-unvalidated (await GIXSGUI), not silently asserted.

**Base corrections — pyFAI bit-exact.** `corrections/stack.py` solid-angle ==
`ai.solidAngleArray()` and polarization == `ai.polarization(factor)` to **maxdiff 0.0** (factors
0.95/0/1/−0.5); `corrections=None` ⇒ weight ≡ 1 exact no-op; factors compose without
double-counting; `weight == 1/normalization`.

**Stitch generator refactor — correct.** `run_stitch`'s `_iter_images`/`frames_factory` build a
**fresh** generator per pass (no one-shot generator consumed twice); `max_eager_bytes` fires on
the `multigeometry` (+ legacy) backend that truly materializes, and is correctly bypassed on the
streaming `pyfai_hist`/GI path (verified empirically); `len(labels)` correctly replaces
`len(images)` incl. `frame_indices` subsets; the fail-loud guards (negative/0/NaN monitor,
image↔integrator length desync, GI αi count) all fire through the streaming path.

**Sources — correct (besides the two findings above).** `CompositeFrameSource` motors cache +
NaN-pad contract + per-call copy; global re-index + label uniqueness; `d090aba` `frame_for`
overrides resolve the *raw* pointer (SpecSource from `_frame_map`; ProcessedNexusSource from the
ORIGINAL master, two hops out — not the reduced `.nxs`) and drop cleanly when absent;
`discover_scans` walk/dispatch/malformed-dir handling.

**Persistence provenance — round-trips.** `StitchPlan.provenance()`/`RSMPlan.provenance()` embed
`corrections`+`gi`; `write_/read_stitched` and `write_/read_rsm` round-trip the blob identically
(1d/2d/rsm); `from_dict` reconstructs the stacks; capability feature-detect (ADR-0002); schema
validates when present; old files (no group) → `None`/`KeyError`-guarded, no crash.

**RSM (re-verified at HEAD).** Two-gridder `Σraw/Σnorm`, mask drops from both sums, empty→NaN
(no 0/0 warnings), streaming ≡ single-shot **with corrections on**, weight wavelength-independent,
header→ai bridge geometrically consistent (solid angle peaks at beam centre). *(My first RSM pass
reviewed a pre-`8addcdd` state and reported the old `Σ(raw·w)`; superseded — the fix verification
above is authoritative.)*

---

## 3. Correctly deferred (real-data-gated; not defects)
These are flagged in `stitching_rsm_build_plan.md` / `design_stitch_rsm_accumulator_jun2026.md`
and gated, not silently asserted — leave as-is until the live/notebook gate:
- GI **absolute** composition signs + `sample_orientation`/`tilt` — pending a GIXSGUI worked-example
  overlay. GI stitch stays on `multigeometry` in the GUI until then (correct conservatism).
- `xu_hist` backend (P3c): the xu q-provider circle order + χ == pyFAI `chiArray` — best validated
  with the real-data notebook, not guessed. `xu_q_frames` is dead-but-proven.
- P7 GUI wiring (stitch viewer, wrangler stitch/GI panels, Refine button).

---

## 4. Remediation plan (prioritized)
1. **[major]** Add the `isfinite` rot1/rot2 guard to `create_multigeometry_integrators` +
   a legacy-path regression test. ~20 min; do before P7 so the multi-source fallback is safe.
2. **[minor]** `discover.py`: top-level `natsort` import (or a one-time warning in the fallback).
3. **[nits]** as convenient: `_partition_selected` raise-vs-widen; fold `xu_q_frames` into the
   shared streaming helpers when P3c lands.
4. **Scientific tail (already tracked):** the GIXSGUI real-data GI-sign gate + P3c circle-order /
   χ gate — the batched convention validations in the build plan's LIVE CHECKLIST.

## 5. Housekeeping (env artifacts, not code)
- A stray empty `tests/core/conftest_xustub.py` and a stale `.git/index.lock` exist from a
  review agent's sandbox; both are inert (the conftest is misnamed so pytest never loads it) but
  the `index.lock` can block git ops. Remove from your Mac: `rm -f .git/index.lock
  tests/core/conftest_xustub.py`.
