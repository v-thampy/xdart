# Handoff — `Diffractometer`: step 4 DONE (live check pending) → 4b/5 next

ADR-0007 §6 **steps 0–4 are implemented and green** on branch `feature/geometry`
(off `feature/gui-redesign-direction-a`, NOT pushed). What remains is a **live
verification** of the step-4 behaviour change, then **4b (`refine_goniometer`)**
and **5 (retire)**.

## State (branch `feature/geometry`, commits `5514a00`→`ed02260`, NOT pushed)

The one shared `Diffractometer` object now drives everything:
- Object + both adapters + presets, `DetectorCalibration`/`ImageOrientation`,
  `from_pyfai_goniometer` (cross-term + moving-base guards), legacy bridges,
  capability-gated persistence — all test-gated incl. the real-data pyFAI gate.
- **psic is the default** (validated `RSM_process.ipynb` values: sample
  `(mu,eta,chi,phi)`, detector `(nu,del)`, `camera=('z-','x-')`,
  `HXRD([0,1,0],[0,0,1])`). A bare `Diffractometer()` is psic-oriented.
- **Step 4 repoint (done):** `Diffractometer` is a byte-equal drop-in everywhere
  (`derive_per_frame`/`make_hxrd`/`init_area_*` aliases): RSM `PixelQMap` + gridding,
  `core/config.diff_config`, `Scan.geometry`, the finish-site blob write (flipped
  on), and the xdart producers (`LiveScan.default_geometry` → psic;
  `_load_from_nexus_v2` prefers the persisted blob).
- Gates: **core 1191 + xdart 1087 passed**; the GI live/batch/reload equivalence
  spine (18) stays green (the repoint kept the 3 GI quantities byte-identical).

## LIVE CHECK (Vivek, `conda activate xrd_test`, the `xdart` console command)

The step-4 change alters live behaviour, so confirm in the app before merge:
1. Run a **psic** scan (live + batch). Per-frame geometry + GI integrate as
   before (no change — derive_per_frame is byte-equal).
2. Confirm the saved `.nxs` now carries `/entry/diffractometer` (the blob) and a
   reload restores `scan.diffractometer` (offline stitch/RSM ready). NeXus viewer
   or `ProcessedScan(path).diffractometer`.
3. **Heads-up:** the app default is now psic, so a **non-psic** scan that never
   sets geometry will not auto-derive `per_frame_geometry` (psic's `nu`/`del` are
   absent). If you run two-circle, set `scan.geometry = Diffractometer.two_circle(...)`
   explicitly (the GUI geometry control is the place to surface this).

## PROMPT (4b → 5)

> Continue ADR-0007 on branch `feature/geometry` (`conda activate xrd_test`; do NOT
> push/bump). Steps 0–4 are done + green.
>
> **Step 4b — `refine_goniometer`** (headless; the Refine-button backend, ships with
> stitch/RSM): a control-point `scipy.least_squares` fit (NOT pyFAI `refine3`, which
> diverges for the stacked psic), seeded from a base `.poni` + calibration images +
> their `(del,nu)` metadata + calibrant; recovers per-axis scale+offset (incl.
> motor-zero offsets) + the detector mount; returns a fitted `Diffractometer` (populates
> the canonical object — the inverse of `from_pyfai_goniometer`). *Gate (real data,
> `~/repos/example_notebooks/Stitching/`):* re-fit the `Multi120_Calibration_*` LaB6
> images → reproduces the saved goniometer params; del-only matches pyFAI `ROBL_v1` beam
> centre < 1 px; recovered `del`/`nu` offsets nonzero; **and validate the azimuthal χ
> independently** (the |q| fit does NOT constrain it — design §3.5, `χ = −atan2(qx, qz)`).
>
> **Step 5 — retire** `DiffractometerConfig`/`DiffractometerGeometry`: they are now only
> kept as the value-preserving reference + the bridges' source/target. Convert remaining
> independent authoring to the one object (or leave thin deprecation aliases). *Gate:*
> grep finds no independent authoring of the two encodings; full suite green.

## Notes / traps

- The repoint is **byte-equal by construction** (the aliases just rename methods the
  consumers already call) — proven by the adapter equality tests + the unchanged suite
  tallies. The only behaviour change is the app **default** (two_circle → psic).
- `circle_motors` is persisted but still not adapter-consumed (RSM passes its motor list
  explicitly); the psic order is now the RSM_process-confirmed `(mu,eta,chi,phi),(nu,del)`.
- `fourc` xu axes remain unvalidated (no fixture) — don't trust its `to_qconversion`.
- **Version floor (maintainer):** the writer now emits the `diffractometer` group
  (additive, back-compat absent→None); a consumer that *requires* it gates on the
  capability.
