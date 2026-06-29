# Stitch / RSM merge-accumulator inconsistency — finding + reconciliation proposal

**Status:** IMPLEMENTED (Jun 2026, maintainer-approved). RSM is unified onto
`Σraw/Σnorm` and the GI footprint is flipped to the boost `1/sin αi`. The pinning
test (`tests/core/test_merge_accumulator_consistency.py`) lost its `xfail` and now
asserts both pipelines recover `true`; full core suite green (1348✓) + the xu
cross-check green (footprint `1/sin` consistent across primitive / GI-stack /
RSM-weight sites; both pipelines recover `true`; xu Fresnel still peaks at αc).
**Still outstanding (independent):** a real-data GIXSGUI overlay for the *absolute*
GI conventions (`sample_orientation`/`tilt`); GI stitch stays on `multigeometry`
in the GUI until that lands.
**Origin:** surfaced while validating the GI footprint correction against
xrayutilities (the GI‑corrections notebook) for wiring GI stitch into the GUI.

---

## 1. The finding

The two reduction pipelines that share `xrd_tools.corrections` weights use **two
different merge accumulators**:

| Pipeline | File / line | Accumulator |
|---|---|---|
| **Stitch** (histogram) | `integrate/stitch_hist.py:138` | `I = Σ raw / Σ norm` |
| **RSM** (3‑D gridder) | `rsm/gridding.py:86` | `I = Σ(raw·w) / Σ w` |

These are **not the same operation**, yet `rsm/gridding.py:51` asserts RSM is
*"the SAME weighted‑mean accumulator as the stitch histogram merge
(stitch_hist.stitch_q_grid)."* That comment is wrong, and both pipelines pull the
**same** per‑pixel weight from `GICorrectionStack.gi_normalization` (and from the
base `CorrectionStack.normalization` — solid angle / polarization). A single
weight value cannot be correct in both schemes.

### Why they differ (the math)

Model a multiplicative correction as `raw = true · C` where `C` is the per‑pixel
*boost* (e.g. solid angle Ω, or the footprint over‑illumination `1/sin αi`). The
goal of the merge is to recover `true`.

- **`Σraw/Σnorm` with `norm = C`:** `I = Σ(true·C) / ΣC` = the `C`‑weighted mean of
  `true`. For constant `true` in a bin this is **exactly `true`** — the correction
  is applied. This is pyFAI's standard convention, and it is what the
  `GICorrectionStack` docstring contracts for:
  *"intensity weights → they multiply into the `Σnorm` denominator … `I = Σ raw / Σ norm`."*
- **`Σ(raw·w)/Σw` with `w = C`:** `I = Σ(true·C·C) / ΣC` = `C`‑weighted mean of
  `true·C` ≠ `true`. **No choice of `w` recovers `true`** from `raw = true·C` in
  this scheme — it can only *weight* frames/pixels, not *apply* a multiplicative
  correction. (To use a weighted mean correctly you must divide the correction
  into the *raw* channel first: feed `raw/C` with a separate reliability weight.)

So the RSM accumulator, fed `(raw·w, w)` with `w` = a correction factor, does **not
correctly apply** that correction — for GI footprint/Fresnel/absorption *or* for
the base solid‑angle/polarization weights. With `w = 1` (no corrections) it
degenerates to `Σraw/Σcounts` (the count‑mean), which is why the default path looks
fine and the regression went unnoticed.

### The footprint symptom

While checking the GI footprint against xrayutilities + the team's
`Multi120_GI_Corrections_Explorer.ipynb`:

- Over‑illumination at grazing boosts measured `raw` by `1/sin αi`, so in the
  **stitch** `Σraw/Σnorm` scheme the weight must be the boost `1/sin αi`. The code
  uses `sin αi` (`grazing.py` `gi_normalization`, footprint branch) — **inverted**,
  off by `1/sin²αi` (×82,000 at αi=0.2°; numerically demonstrated). The
  `test_grazing.py` module docstring even states the intended gate "footprint ∝
  1/sin αi", which the code contradicts.
- But flipping it to `1/sin αi` **breaks the RSM footprint test**
  (`test_rsm_corrections.py::test_footprint_only_is_sin_alpha_i`), because RSM's
  different accumulator wants the opposite direction. The footprint sign is a
  *symptom* of the accumulator split, not an independent bug.

Fresnel (`norm = |T(αi)|²|T(αf)|²`) and the absorption path (`norm = 1/P`) happen
to be in the boost direction already, so they're correct for `Σraw/Σnorm` and only
the footprint reads as inverted there.

---

## 2. Proposal — unify both pipelines on `Σraw / Σnorm`

`Σraw/Σnorm` is the correct convention for multiplicative corrections, is pyFAI's
own, and is the contract the `CorrectionStack` / `GICorrectionStack` already
document. Make RSM match it.

1. **RSM gridder** (`rsm/gridding.py`): `_feed_pair` accumulates `Σraw` (not
   `Σ(raw·w)`) in the raw channel and `Σnorm` in the norm channel; `_pair_intensity`
   stays `num/den`. i.e. feed `(img, w)` not `(img·w, w)`. Behaviour‑preserving for
   the no‑correction default (`w = 1` ⇒ `Σraw/Σcounts`, unchanged). Update the
   `:51` comment to state the convention correctly.
2. **Footprint direction** (`grazing.py` `gi_normalization`): footprint
   `norm ·= 1/sin αi` (the boost), so `corrected = raw·sin αi` — matching the
   notebook, textbook GIXS, and the now‑consistent `Σraw/Σnorm` direction.
3. **Tests:** flip the footprint assertions in `test_grazing.py`,
   `test_stitch_gi.py`, `test_rsm_corrections.py` to the boost direction; the
   pinning `xfail` (below) becomes `xpass` → promote to a plain assertion.
4. **Validation:** re‑run the xu cross‑check; then a real‑data GIXSGUI overlay for
   the absolute GI conventions (`sample_orientation`/`tilt`) — still outstanding,
   independent of this fix.

### Risk / blast radius
- Changes the numbers produced by **any** corrected RSM reduction (GI *and* base
  solid‑angle/polarization). The no‑correction default is unchanged. Anyone who
  tuned to the current (incorrect) RSM corrected output will see a shift — this is
  a correctness fix, call it out in the changelog.
- Stitch's corrected output also shifts (footprint flip) — same nature.
- Do **not** land piecemeal: flipping footprint alone (without unifying RSM) leaves
  the two pipelines knowingly inconsistent. Unify the accumulator + the footprint
  together.

---

## 3. The pinning test

`tests/core/test_merge_accumulator_consistency.py` exercises the *actual* RSM
accumulator (`rsm.gridding._feed_pair`/`_pair_intensity`) on a known
`raw = true·C` and asserts it recovers `true`. It currently **xfails** (RSM gives
`Σ(true·C²)/ΣC ≠ true`); after the unification (step 1) it will **xpass** — at which
point drop the `xfail` marker. A companion check shows `stitch_q_grid` already
recovers `true`, isolating the defect to the RSM accumulator.

---

## 4. Why this is the maintainer's call

It changes committed, tested headless physics in two pipelines and shifts every
corrected reduction's numbers. The reconciliation direction (`Σraw/Σnorm`) is
well‑founded, but the decision to re‑baseline corrected output — and the real‑data
GIXSGUI revalidation of the absolute GI conventions — belongs with the maintainer.
GI stitch stays on `multigeometry` in the GUI until this is resolved.
