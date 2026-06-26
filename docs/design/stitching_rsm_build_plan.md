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
- **P3c — `xu_hist` backend** (the design default; deferred): the xu q-provider
  (`to_qconversion` → `Ang2Q.area`) + the per-frame sample-angle assembly from
  `circle_motors` (the "one wiring task"). *Gate:* xu_hist |q| == pyfai_hist within the
  radial bin width (validates the psic circle order) + χ == pyFAI `chiArray`. Best done
  with the real-data notebook so the circle order is validated, not guessed.
- **P4 — GI flag** (`StitchPlan.gi: GIMode | None`, reusing the reduction `GIMode`/`gi_config`;
  see `design_stitching_jun2026.md §2.8`). GI orthogonal to backend; no `Diffractometer`
  extension needed (the one wiring task is the per-frame sample-angle assembly from
  `circle_motors`).
- **P5 — persistence**: register `stitched_1d/2d` schema groups + capability (mirror the
  `diffractometer` group), persist the applied-`CorrectionStack` + plan provenance.
- **P6 — RSM**: unify the RSM pipeline onto the one `Diffractometer` + the shared accumulator +
  the `CorrectionStack` weight.
- **P7 — [LIVE-GATED] GUI**: Stitch viewer controller + layout, the wrangler stitch/GI panels,
  the Refine button (wrapping `refine_goniometer`). Thin, isolated; the only part that waits.

## LIVE CHECKLIST (run when `xrd_test` GUI is available; each maps to a commit)

- [ ] Geometry step-4: a psic scan live+batch integrates as before; the saved `.nxs` carries
      `/entry/diffractometer`; reload restores `ProcessedScan.diffractometer`. (app default is
      now psic — a non-psic scan must set `scan.geometry` explicitly.)
- [ ] P1: a live stitch reproduces the notebook stitch.
- [ ] P7: the Stitch viewer + Refine button + GI stitch flow.

## Status
Geometry (ADR-0007 steps 0–5 + 4b `refine_goniometer`) — **done, reviewed, green**.
**P1, P2a, P2b, P3a, P3b — done + gated.** Next: **P3c** (`xu_hist` backend — the design
default; needs the `circle_motors` angle-assembly validated, ideally with the real-data
notebook), then P4 (GI flag), P5 (persistence), P6 (RSM), P7 (live GUI). The live tail is P7.
