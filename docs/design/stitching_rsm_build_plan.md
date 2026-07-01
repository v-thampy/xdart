# Build plan — stitching + RSM without live gating (headless-first)

**Context (Jun 2026):** live GUI testing in `xrd_test` is intermittently unavailable, so
we build the stitching + RSM modules **headless-first** and quarantine the live-gated GUI
wiring into a thin tail. This doc is the durable roadmap + the fallback map (each phase ends
at a green, committed state) and the running **LIVE CHECKLIST**.

## Working practices (the substitute for live gating at intervals)

1. **Real-data fixture gates = the live proxy.** Vendor small real fixtures into the repo and
   assert headless output against a known-good reference (pyFAI arrays, the notebooks'
   `stitched_*.xye` / saved gonio / an RSM slice). Precedent: the vendored `gonio_*.json` +
   the pyFAI `integrate1d` correction gate.
2. **Equivalence spine.** Build each piece so **headless == reload** (== live, eventually) by
   construction — so a future live divergence is a real bug pointing at a specific commit.
3. **Dead-but-proven core, live-gated wiring** (ADR-0006 pattern). Land the headless engine
   fully tested but *inert* (no GUI change) — it ships without a live checkpoint. The GUI
   wiring is the only commit that waits.
4. **Small themed commits + a running LIVE CHECKLIST.** Each commit is a fallback point mapped
   to what to verify live. **Adversarial review at each phase boundary** (it caught the `rot3`
   divergence + the conditioning collapse headlessly — the "live shook it out" class of bug).

## Phases (each ends green + committed)

- **P1 — stitch consumes the `Diffractometer` ✓ DONE** (`integrate/multi.py` +
  `analysis/plans.py`, `8a1347a`): `create_multigeometry_integrators_from_geometry` uses
  `to_pyfai_per_frame` (fitted scales, GAP A) over a `DetectorCalibration` (Detector_config,
  GAP B); `run_stitch` dispatches on `StitchPlan.diffractometer` (legacy deg2rad = fallback).
  Gated: uncalibrated psic == legacy end-to-end; calibrated == `get_ai`. *Live gate:* the
  notebook `stitched_LaB6_17keV_scan14.xye`.
- **P2a — shared correction stack ✓ DONE** (`corrections/stack.py`, `937a978`): the per-pixel
  normalization (solid-angle/polarization) at the accumulator seam, `Σraw/Σnorm` == pyFAI.
- **P2b — GI corrections ✓ DONE** (`corrections/grazing.py`, `7542c0e`): footprint/refraction/
  Fresnel/absorption from xu materials; the INTENSITY-vs-POSITION split (refraction →
  `refract_q`, the rest → `gi_normalization`). Gated notebook-free (Si@10keV: Fresnel peaks at
  αc; refraction shift vanishes above αc; footprint ∝ 1/sin αi). *Flagged:* the footprint +
  path-absorption composition signs are convention-dependent — verify vs a GIXSGUI worked
  example with live data.
- **P3a — histogram merge ✓ DONE** (`integrate/stitch_hist.py`, `1f37cb3`):
  `stitch_q_grid` (streaming `Σraw/Σnorm` over (q[,χ]) bins) + the `pyfai_hist` provider
  (`pyfai_q_frames`). Gated: single-frame == pyFAI `integrate1d` exact; multi-frame ==
  MultiGeometry shape within 3%.
- **P3b — `StitchPlan.backend` dispatch ✓ DONE** (`analysis/plans.py`, `ae754e1`):
  `multigeometry` | `pyfai_hist` routed in `run_stitch`; `xu_hist` raises (deferred).
- **P3 adversarial review ✓ DONE** (10 confirmed; `wgtp49v9r`): fixed the silent-corruption
  class in `pyfai_q_frames` — a bad per-frame monitor (0/NaN/**negative** → sign-flip cancels
  healthy frames) and a `zip(images, integrators)` length desync (silent truncation) now both
  fail loud; mirrored the negative-monitor guard into MG `_prepare_images`. The `pyfai_hist`
  dispatch rejects a non-q `unit` (provider emits q Å⁻¹ only) + leftover `extra` pyFAI kwargs,
  and warns on an ignored `method`. + merge-level + dispatch regression tests (monitor/NaN/
  empty-bin/mask-merge/2D-seam/2D-npt/corrections=None).
- **P3c — `xu_hist` backend** (the design default; deferred): the xu q-provider
  (`to_qconversion` → `Ang2Q.area`) + the per-frame sample-angle assembly from
  `circle_motors` (the "one wiring task"). *Gate:* xu_hist |q| == pyfai_hist within the
  radial bin width (validates the psic circle order) + χ == pyFAI `chiArray`. Best done
  with the real-data notebook so the circle order is validated, not guessed.
- **P4 — GI flag ✓ DONE (headless; convention live-gated)** (`StitchPlan.gi: GICorrectionStack`,
  `pyfai_gi_q_frames`): GI on the `pyfai_hist` backend only. The per-pixel αf + out-of-plane
  q_z come from **pyFAI's own fiber units** (`FiberIntegrator` + `exit_angle_vert`/`qoop`
  after `reset_integrator(incident_angle=…)`) — the SAME convention as the reduction GI path,
  gate-pinned by `q_oop ≡ k0·(sin αf + sin αi)` (`test_stitch_gi.py::TestGIConvention`). The
  P2b `GICorrectionStack` then supplies the weight (footprint·Fresnel·absorption → `Σnorm`)
  + refraction (→ the q-map). Per-frame αi = `StitchPlan.gi_incident_angle_deg` else the
  `Diffractometer.incident_angle` mapping. **Gated headless:** GI-off ≡ non-GI; footprint-only
  ⇒ `I = I_nonGI / sin αi`; refraction toggle; backend/diffractometer guards. **NOT validated:**
  the absolute composition signs (P2b flag) + `sample_orientation`/`tilt` — pending GIXSGUI.
- **P4+P5 adversarial review ✓ CLEAN** (`w39j3h9ol`, worktree-isolated, 3 dimensions ×
  verify): 0 confirmed findings. Finders ran live reproduction probes for the FiberIntegrator
  cache-leak across frames, refraction q/q_z consistency, the dispatch guards, and the P5
  provenance round-trip — none broke. Main tree verified clean (worktrees auto-pruned).
- **P5 — persistence ✓ DONE** (`io/schema.py` + `io/nexus.py`): registered `stitched_1d/2d`
  as schema groups + capabilities (mirroring the `diffractometer` group — optional, feature-
  detected via `detect_capabilities`, schema-validated when present). `write_stitched` now
  stamps a `provenance_json` vlen-UTF8 blob (the `StitchPlan` + applied `CorrectionStack`/
  `GICorrectionStack` via the new `StitchPlan.provenance()`); `read_stitched` parses it back
  onto the `xr.Dataset.attrs`. Round-trip + capability + validation gated headless. The
  binary stitch pattern already round-tripped; P5 adds the registry + provenance.
- **P6 — RSM (in progress)**: unify the RSM pipeline onto the one `Diffractometer` + the shared
  accumulator + the `CorrectionStack` weight. Design: the Understand-workflow synthesis
  (`wpc9exe8j`). RSM was a 3D streaming **count-mean** gridder (`xu.Gridder3D` NO_NORMALIZATION,
  `Σraw/Σcounts`) with a parallel geometry path + ZERO corrections.
  - **P6.1 ✓ DONE** (`1f9d8d8`): the two-gridder `Σ(raw·w)/Σ(w)` accumulator — `StreamingGridder`
    + `grid_img_data` now run two `xu.Gridder3D` with `Normalize(False)` (bare SUM): `_grid_raw`
    = Σ(raw·w), `_grid_norm` = Σ(w), volume = the ratio. The SAME accumulator as `stitch_q_grid`.
    `weight=1` reproduces the prior count-mean exactly (behaviour-preserving). Good-mask drops
    non-finite-q / non-finite-img / w≤0 from BOTH sums (a NaN coord would be clamped; a masked
    pixel's weight would bias the denominator — both verified on xu 1.7.12).
  - **P6.2 ✓ DONE** (`5fd8c35`): `RSMPlan.corrections` + `rsm/corrections.py` — `detector_header_
    to_ai` (xu mm header → fixed-lab pyFAI ai; gate: solid angle peaks at the beam centre) +
    `rsm_correction_weight` (the same `CorrectionStack.normalization` stitch uses; wavelength-
    independent, computed once; geometry-static). Threaded run_rsm → pipeline → `add(weight=)`.
  - **P6.3 ✓ DONE** (`f5bae7a`): `assemble_circle_angles` (`circle_motors` → per-frame xu angle
    vector) — the shared blocker for RSM-GI + the xu_hist stitch. Convention CARRIED in the
    preset's `circle_motors`, not invented; gate = byte-equal to the legacy explicit-`diff_motors`
    path + q-identity through `pixel_q`. (Production reroute deferred — convention-gated.)
  - **P6.4 ✓ DONE** (`f79806b`): the canonical `Diffractometer` is a byte-equal `PixelQMap`
    drop-in (`from_diffractometer_config` identity through `pixel_q`, every convention knob).
  - **P6.5 ✓ DONE** (`7607787`): `xu_q_frames` provider (dead-but-proven → P3c is pure wiring).
    `|q|` gate-checked; χ azimuth flagged pending the P3c real-data gate (== pyFAI chiArray).
  - **P6.6 ✓ DONE** (`83a51b7`): the equivalence spine — streaming ≡ single-shot + chunk-
    invariant, WITH corrections on (the cases never compared on real intensities before).
  - **P6.7 ✓ DONE** (`c50fcb6`): `RSMVolume ↔ NeXus` (`write_rsm`/`read_rsm`) + the `rsm` schema
    group/capability + provenance, mirroring the P5 stitched persistence.
  - **P6.8 ✓ DONE (intensity weight; refraction live-gated)** (`71c26a2`): `RSMPlan.gi` +
    `gi_grid_weight` — the GI footprint/Fresnel/absorption weight, reusing P4's pyFAI-fiber αf
    (convention-pinned). FIXED-incidence only; **refraction (qz rewrite in 3-D) + per-frame αi +
    the absolute GI signs are the real-data-gated tail** (batch with P3c/P4 GI validation).
  - **All 8 sub-steps committed, core 1301✓.** Remaining for P6: the production angle-assembly
    reroute + the GI refraction/convention — all real-data-gated (the live tail), batched with
    P3c + the P4 GI-sign validation.
- **P7 — [LIVE-GATED] GUI**: Stitch viewer controller + layout, the wrangler stitch/GI panels,
  the Refine button (wrapping `refine_goniometer`). Thin, isolated; the only part that waits.

## LIVE CHECKLIST (run when `xrd_test` GUI is available; each maps to a commit)

- [ ] Geometry step-4: a psic scan live+batch integrates as before; the saved `.nxs` carries
      `/entry/diffractometer`; reload restores `ProcessedScan.diffractometer`. (app default is
      now psic — a non-psic scan must set `scan.geometry` explicitly.)
- [ ] P1: a live stitch reproduces the notebook stitch.
- [ ] P4: a GI stitch vs a GIXSGUI-worked example — confirm the composition signs
      (footprint/absorption direction) + `sample_orientation`/`tilt` against real data.
      (αf/q_z maps are pyFAI's, already pinned; only the absolute correction direction waits.)
- [ ] P7: the Stitch viewer + Refine button + GI stitch flow.

## Status
Geometry (ADR-0007 steps 0–5 + 4b `refine_goniometer`) — **done, reviewed, green**.
**P1, P2a, P2b, P3a, P3b, P3-review, P4, P5, P6 (all 8 sub-steps) — done + gated** (core 1301✓).
P6 unified RSM onto the one `Diffractometer` + the shared Σ(raw·w)/Σ(w) accumulator + the
`CorrectionStack` (+ the shared `assemble_circle_angles` + `xu_q_frames` that make P3c pure
wiring + RSM persistence + the GI intensity-weight seam). **The real-data gate now batches:**
P3c (`xu_hist`) circle order, the P4 GI-sign validation, the xu_q_frames χ azimuth (== pyFAI
chiArray), and the P6 GI refraction/convention + production angle-assembly reroute — all need
the notebook to confirm a geometry convention. Next headless: **P7 (live GUI)** — the only
remaining tail besides the batched convention validations.
