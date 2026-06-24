# Next-session handoff — `Diffractometer` step 4 (repoint consumers) + 4b/5

The headless geometry module (ADR-0007 §6 **steps 0–3**) is **done, validated, reviewed,
and green** on branch `feature/geometry` (off `feature/gui-redesign-direction-a`, NOT
pushed). What remains is the **invasive consumer rewire** (step 4) + `refine_goniometer`
(4b) + retire (5). Step 4 touches the writer + RSM + reduction/GI, so unlike steps 0–3 it
**needs live testing** (the GI guardrail + the live/batch/reload equivalence spine).

## State (branch `feature/geometry`, commits `5514a00`→`0b0858d`, NOT pushed)

Implemented + test-gated (core 1187 passed, 4 skipped):
- `core/geometry/diffractometer.py`: **`Diffractometer`** (frozen) with both adapters
  (`to_pyfai_per_frame` == `derive_per_frame`; `to_qconversion`/`to_hxrd` == `make_hxrd`,
  proven byte-equal with real xu), presets (`two_circle`/`fourc`/`psic`/`sixc`/
  `psic_halpha`), **`DetectorCalibration`** + **`ImageOrientation`**,
  **`from_pyfai_goniometer`** (cross-term + moving-base guards; reproduces pyFAI
  `get_ai` to 1e-12 on the vendored `tests/core/fixtures/gonio_robl_v{1,2}.json`), and the
  **legacy bridges** `from/to_diffractometer_geometry|config`.
- Capability-gated persistence: `io/schema.py` (`diffractometer` GroupSchema + cap +
  `ub_matrix` cap), `io/nexus.py::write_diffractometer`, `io/read.py::get_diffractometer`
  + `ProcessedScan.diffractometer` (None on old files). Finish-site write is wired but
  **gated on `isinstance(geom, Diffractometer)`** — a no-op until step 4 threads the full
  object in (avoids a misleading partial blob).
- ADR-0007 (`docs/decisions/0007-...`). Env: `conda activate xrd_test`; tests
  `python -m pytest tests/core` + `QT_QPA_PLATFORM=offscreen python -m pytest tests/xdart`.

## PROMPT

> Continue ADR-0007 / `design_diffractometer_geometry_jun2026.md` §6 on branch
> `feature/geometry` in `~/repos/xrd-tools` (`conda activate xrd_test`). Steps 0–3 are
> done + green; do **step 4 (repoint consumers)**, then 4b/5. Do NOT push / bump versions.
> Keep the suite green; the GI path needs a live xdart check (Vivek's env).
>
> **Step 4 — repoint every consumer onto the one `Diffractometer`** (the exact sites,
> from the consumer map):
>
> 1. **`Scan.geometry` is duck-typed `Any`** (`core/scan.py:248`), read **once** at
>    `reduction/core.py:587-614` (finish) and consumed only via `.all_referenced_motors()`
>    + `.derive_per_frame(motors)` inside `write_per_frame_geometry` (`io/nexus.py:2052-
>    2064`). `Diffractometer` already has `all_referenced_motors` + `to_pyfai_per_frame`
>    but **not** `derive_per_frame` — so the cleanest drop-in is to add a `derive_per_frame
>    = to_pyfai_per_frame` alias on `Diffractometer` (or have `write_per_frame_geometry`
>    prefer `to_pyfai_per_frame`). Then `Scan.geometry` can be a `Diffractometer` and the
>    finish-site `write_diffractometer` (already `isinstance`-gated) flips on for real.
> 2. **RSM** reaches geometry ONLY through `PixelQMap.diff_config.make_hxrd`
>    (`pixel_q.py:241-309`); the sole xu construction site is
>    `DiffractometerConfig.make_hxrd` (`diffractometer.py:60-83`). Repoint by giving
>    `PixelQMap` a `Diffractometer` (or making `diff_config` a `Diffractometer` view) and
>    calling `diff.to_hxrd(energy)` / `diff.to_qconversion()` — byte-equal, already proven.
>    Watch the scout path `_corner_pixel_q` (`rsm/gridding.py:128-173`) which reuses
>    `mapper.diff_config` with a synthetic header — the scout q-bounds set the final grid
>    axes, so they must stay byte-stable.
> 3. **`core/config.diff_config`** (`config.py:44,76-78`, type `DiffractometerConfig`,
>    read only via `ExperimentConfig.mapper → PixelQMap`) → a `Diffractometer`.
> 4. **xdart producers** that build the object landing in `Scan.geometry`:
>    `LiveScan.default_geometry()` (`xdart/modules/ewald/scan.py:705-721`, currently
>    `DiffractometerGeometry.two_circle`) + `_load_from_nexus_v2` (`:816-827`, `from_json`)
>    + `image_wrangler_thread.py:2584`. Switch these to build a `Diffractometer` (the
>    legacy bridges make this mechanical), so the object handed to the core
>    `Scan`/`NexusSink` exposes the new API and the diffractometer blob actually persists.
> 5. **GI GUARDRAIL (do not violate):** the three GI-defining quantities — per-frame
>    `incident_angle`, `gi.tilt_angle`, `gi.sample_orientation` — must stay **byte-
>    identical**. `_resolve_gi_incident_angle` (`reduction/core.py:2154-2182`) and the
>    FiberIntegrator built from the first frame's incidence (`fiber()`, `:2039-2056`) with
>    per-frame incidence applied in `_run_gi_1d/2d` (`:2417-2499`) must be unchanged.
>    `incident_angle` is in **degrees** (NOT `deg2rad`'d); `rot1/2/3` in **radians**.
>    *Gate:* full suite + the RSM synthetic test + GI matrix unchanged + the
>    `test_*_publication_live_batch_reload_equivalence` spine; **then a live xdart GI run**.
>
> **Step 4b — `refine_goniometer`** (the Refine-button backend, ships with stitch/RSM, not
> the core refactor): a headless control-point `scipy.least_squares` fit (NOT pyFAI
> `refine3`, which diverges for the stacked psic), seeded from a base `.poni` + calibration
> images + their `(del,nu)` metadata + calibrant; recovers per-axis scale+offset (incl.
> motor-zero offsets) + the detector mount; returns a `Diffractometer` (populates the
> canonical object). *Gate (real data):* re-fit the `Multi120_Calibration_*` LaB6 images
> (`~/repos/example_notebooks/Stitching/`) → reproduces the saved goniometer params;
> del-only matches pyFAI `ROBL_v1` beam centre < 1 px; recovered `del`/`nu` offsets nonzero;
> **and** validate the azimuthal χ independently (the |q| fit does NOT constrain it — see
> design §3.5, `χ = −atan2(qx, qz)` in `HXRD([0,1,0],…)`).
>
> **Step 5 — retire** `DiffractometerConfig`/`DiffractometerGeometry` (or leave deprecation
> aliases) once all callers move. *Gate:* grep finds no independent authoring of the two
> encodings; full suite green.

## Notes / traps

- **Drop-in seam:** `Diffractometer` needs a `derive_per_frame` alias to be a transparent
  `Scan.geometry` replacement — that's the single smallest step-4 change that flips the
  persistence on.
- **Byte-equality is already proven** for both adapters (`test_diffractometer.py`), so
  step 4 is value-preserving by construction — the risk is the *plumbing*, not the math.
- **`circle_motors`** is carried but not yet adapter-consumed; if RSM starts feeding angles
  from it (instead of an explicit `diff_motors` list) cross-validate the psic
  sample-circle↔motor order against real `Ang2Q.area` usage first.
- **Version floor (maintainer):** once step 4 makes the writer emit/consume the
  `diffractometer` group, the release floor must cover it (additive + back-compat, so an
  old reader just sees no group; a consumer that *requires* it gates on the capability).
- **`fourc` xu axes are unvalidated** (no fixture) — don't trust its `to_qconversion`
  until a real four-circle calibration validates it.
