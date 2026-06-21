# Azimuthal Mode A (non-GI I-vs-χ) — fix the run/reintegrate path + persist

**Date:** 2026-06-20 · **Status:** designed, ready to implement (its own focused effort).
**Goal:** make the non-GI azimuthal profile (Mode A: I vs χ, `chi_deg`) actually
work when run/reintegrated, and persist it round-trippably. Vivek: PERSIST (with a
writer axis-kind guard).

## The bug (Round-38 review, CONFIRMED)
Mode A is selectable — the Int-1D axis combo's third entry "χ (°)" sets
`unit='chi_deg'` (integrator.py:1847; `Units_dict` maps it; integrator.py:2162).
The headless wrapper `xrd_tools.integrate.single.integrate_radial` (single.py:105)
is correct (pooled count-weighted I(χ) over a q-band, output axis χ) AND the
LEGACY per-frame path already dispatches to it — `frame.integrate_1d` has a
`if str(unit) == 'chi_deg': … integrate_radial(radial_range=<q-band>, …)` branch
(frame.py:619-632).

BUT the **run / reintegrate spine** (`xrd_tools.reduction.core._reduce_frame`,
the non-GI `else` branch, core.py ~2171-2188) has **no `chi_deg` branch** — it
calls `integrate_1d(image, ai, unit='chi_deg', radial_range=<q-band>, …)`
unconditionally. pyFAI's `integrate1d` then treats `radial_range` (a q-band) as a
**χ output range** → a garbage sliver — and the reducer persists it unguarded
into `integrated_1d` with `units="chi_deg"`. So a live scan / reintegrate in Mode
A produces + saves wrong data. (Mode B, GI `CHI_GI`, is correctly routed through
`_run_gi_1d` (core.py:2365, 2414) — only Mode A is broken.)

## The fix
### 1. Run-path dispatch (the core spine)
In `_reduce_frame` non-GI branch (core.py ~2171), BEFORE the `integrate_1d` call,
add a `chi_deg` dispatch mirroring `frame.integrate_1d:619-632`:
- if `plan.integration_1d.unit == 'chi_deg'`: call
  `integrate_radial(image, ai, npt=plan.integration_1d.npt,
   radial_range=plan.integration_1d.radial_range,  # the q (or 2θ) BAND
   radial_unit=<the q/2θ unit, NOT chi_deg>, mask=mask,
   polarization_factor=…, normalization_factor=_normalization_for(...))`.
  Note `integrate_radial` does NOT accept `error_model`/`azimuth_range` the same
  way — see the legacy branch for the exact kwargs it drops.
- else: the existing `integrate_1d(...)`.
- **Param-mapping is the crux:** for Mode A the OUTPUT axis is χ, so
  `integration_1d.radial_range` must carry the **q-band to integrate over** and
  `npt` the **χ-bin count**, with the *radial_unit* being the q/2θ unit (where
  does Mode A's radial_unit come from? — the GUI currently only has the one
  axis-unit combo set to chi_deg; confirm how the legacy path sources radial_unit
  (likely a fixed q default) and replicate). Get this mapping right or you
  reproduce the bug with a different sliver.

### 2. Plan / GUI plumbing
- Verify `plan_from_live_scan` carries `unit='chi_deg'` + the q-band into
  `Integration1DPlan` for Mode A (it likely passes the GUI's bai_1d_args through;
  check the radial_range field maps to the q-band, not a χ-range).
- The Int-1D range fields under Mode A should mean the q-band (label them so);
  the χ output span is ±180 (integrate_radial's default) unless a field is added.

### 3. Persist + writer axis-kind guard (Vivek: persist)
- The `chi_deg` 1D result (unit='chi_deg', χ axis) must round-trip in
  `integrated_1d` like Mode B does for `chi_gi`. Check `FrameView`/writer axis
  handling (`two_d_kind_from_units` / the 1D axis-kind) classifies `chi_deg`
  correctly and the reader returns it with the right unit/label.
- Add a **writer axis-kind guard**: a 1D row whose unit is an azimuthal kind
  (`chi_deg`/`chigi_deg`) must be written/validated as such — never silently
  persisted as a q-axis row. Do NOT loosen the uniform-axis validators; this is an
  additive correctness check (reject/skip a mis-axis'd row per-frame, fail-loud).

## Verification gates (must pass before merge)
- **Live check:** select χ (°) in Int-1D on any scan → a correct I-vs-χ curve
  (smooth azimuthal profile), NOT a wrong/empty sliver.
- **Round-trip:** save → reload → the χ curve + `chi_deg` unit/label survive.
- **Equivalence spine:** add/extend a Mode-A case to
  `test_gi_batch_real_data` so live ≡ batch ≡ reload for a `chi_deg` 1D.
- **Reintegrate-1D in χ:** Reintegrate 1D with Mode A selected → correct + saved.
- **Mode B unaffected:** GI χ_GI still routes through `_run_gi_1d` (regression).
- **Headless parity:** the spine's Mode-A result matches the legacy
  `frame.integrate_1d` chi_deg result for the same frame.

## Notes
- Keep `display_logic` Qt-free (purity guard) if you touch axis classification.
- Label the bug **pre-existing** (the spine never had the chi_deg branch; the
  feature shipped with only the legacy + headless paths wired).
