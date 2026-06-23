# Next-session handoff — shared `Diffractometer` geometry (headless, no live test)

Paste the **PROMPT** block into a fresh session to start the geometry module. It is
the gating prerequisite for stitching + RSM and is **fully headless** — its gates are
unit tests + the real-data LaB6 notebook, so it can be built while the app can't be
live-tested.

---

## State (branch `feature/gui-redesign-direction-a`, base commit `8eb69ac`, NOT pushed)

The Scan Plotter (ROI + right axes), SPEC source, and the shared `ScanSourceWidget`
+ headless `CompositeFrameSource`/`parse_scan_groups`/`discover_scans` are done +
green (core 1111 / xdart 1087 offscreen), pending live-test. Geometry is independent
of those, but stitching (next) will reuse `CompositeFrameSource`, so branch geometry
**off `8eb69ac`** to keep the lineage.

- **Design (source of truth):** `docs/design/design_diffractometer_geometry_jun2026.md`
  — read it IN FULL first. §2 = what exists; §3.2 = the layered shape; §3.2a =
  `from_pyfai_goniometer`; §3.4–3.5 = the validated fit recipe; §4 = persistence; §6 =
  the gated step sequence. Promote to **ADR-0007** when ratified.
- **Consumer:** `design_stitching_jun2026.md` §2.5 (GAPs A–F) + §3.1–3.2a — geometry
  closes GAPs A–D (hardwired deg2rad; PONI drops Detector_config; no full-geometry
  input path; no goniometer importer).
- **Exists:** `core/geometry/diffractometer.py` (`Diffractometer` + `AngleMapping`),
  `core/geometry/pixel_q.py`. **Missing:** `DetectorCalibration` (PONI + detector_config
  + image-orientation transform), `from_pyfai_goniometer`, `stitch_ponis(images,
  geometries)`, the schema/persistence, the consumer rewiring.
- **Real-data fixture (the gate):** `examples/.../Stitching/stitch_simplified.ipynb`
  + `MG_gonio_object.json` (pyFAI GoniometerRefinement: Pilatus 300k-w on a psic arm,
  LaB6 17 keV, del/nu mesh). The fitted goniometer is the production instance (§3.4).

Env: `conda activate xrd_test`. Tests: `python -m pytest tests/core` (headless) and
`QT_QPA_PLATFORM=offscreen python -m pytest tests/xdart`. Do NOT push / bump versions.
Do the geometry refactor test-first; it touches RSM + the writer — keep them green.

---

## PROMPT

> Build the shared `Diffractometer` geometry module (headless) on a branch off
> `8eb69ac` in the xrd-tools monorepo (`~/repos/xrd-tools`; activate `xrd_test`).
> Read `docs/design/design_diffractometer_geometry_jun2026.md` IN FULL first (and
> `design_stitching_jun2026.md` §2.5 GAPs A–F) — it is the source of truth and should
> be promoted to ADR-0007. Work the §6 gated step sequence, committing + testing each
> step (headless `tests/core`; the real-data gate is the LaB6 notebook +
> `MG_gonio_object.json`, NOT live GUI testing):
>
> 1. **Layer the geometry object** (§3.2): add `DetectorCalibration` (PONI fields +
>    `detector_config` + the image-orientation transform, closing stitch GAP B/E) and
>    keep `Diffractometer` as the per-frame `rot1/2/3` description carrying the FITTED
>    per-axis `AngleMapping(sign=scale, offset)` (not a hardwired deg2rad — GAP A).
>    *Gate:* round-trip JSON; preset-consistency.
> 2. **pyFAI goniometer bridge** (§3.2a): `Diffractometer.from_pyfai_goniometer(json)`
>    parsing the `GeometryTransformation` expressions → `AngleMapping`s + base
>    `DetectorCalibration`; plus `to_qconversion()` / `to_hxrd(energy)`. *Gate (real
>    data):* load `MG_gonio_object.json`, reproduce pyFAI per-frame rot1/2/3.
> 3. **`stitch_ponis(images, geometries)`** (stitching §3.2, GAP C): accept a list of
>    complete per-frame `DetectorCalibration`s → `MultiGeometry`; make
>    `create_multigeometry_integrators` the thin "base ⊕ Diffractometer → per-frame
>    geometries" feeder (built with explicit `unit=` + mandatory detector mask — GAP
>    F). *Gate (real data):* stitch the notebook LaB6 mesh, assert it matches the
>    pyFAI-`Goniometer` reference within the radial bin width.
> 4. **Persist** (§4, schema-as-code, additive + capability-gated): write the
>    `Diffractometer`/`DetectorCalibration` (+ UB as a separate capability) through the
>    sink; expose `scan.diffractometer` on the reader. *Gate:* write→read round-trip;
>    back-compat (absent → None).
> 5. **Rewire consumers + retire** the parallel representations (RSM `qconversion`,
>    `core/config.diff_config`) to the one object; leave aliases if needed. *Gate:*
>    full suite green; grep finds no independent geometry rep.
>
> The fit recipe (§3.5) is validated (del/nu mesh + del-only cross-check); the Refine
> button is a later thin GUI wrapper — keep `refine_goniometer` headless. After
> geometry lands, the next headless module is **stitching** (its source/grouping half
> is already built via `CompositeFrameSource`); then RSM/fitting. Defer all GUI/viewer
> wiring until live testing is possible.
