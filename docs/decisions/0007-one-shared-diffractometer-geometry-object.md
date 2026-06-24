# ADR-0007: one shared `Diffractometer` geometry object (kill parallel representations)

**Status:** accepted (headless core, steps 0–3) · 2026-06-23 · promotes
`docs/design/design_diffractometer_geometry_jun2026.md`.
**Builds on:** ADR-0002 (capability attrs — the persistence mechanism).
**Consumed by:** `design_stitching_jun2026.md` (all three stitch backends) and
`design_rsm_jun2026.md` — both reference this object as their single geometry input.
**Scopes against:** the guardrail "do not force-merge two genuinely-different derived forms."

## Context

Two objects in `core/geometry/diffractometer.py` encoded the *same physics* in two
consumer-specific conventions, authored independently → they could silently drift:

- **`DiffractometerConfig`** — the xrayutilities / RSM view (ordered circle stacks
  `sample_rot`/`detector_rot`, `r_i`, HXRD refs, camera orientation); `make_hxrd(energy)`
  builds `xu.QConversion` + `xu.HXRD`. Consumed by RSM `PixelQMap` / gridding.
- **`DiffractometerGeometry`** — the pyFAI / integration view (per-axis `AngleMapping`s
  → `rot1/rot2/rot3` (rad) + GI `incident_angle` (deg)); `derive_per_frame(motors)`.
  Consumed by reduction / GI / stitch and persisted as `/entry/per_frame_geometry`.

The two have **no shared fields** and **independently authored presets**; the
`two_circle`/`psic` correspondence between them lived only in a human's head. The
stitching notebook (real LaB6 + Pilatus-300k-w on a psic arm) also surfaced three
concrete gaps the current model could not express: a **fitted** per-axis scale (not a
hardwired `deg2rad`), the detector `Detector_config` mount (a bare `PONI` drops it), and
the beamline raw-array image orientation. And a calibrated goniometer in the wild is a
pyFAI `GoniometerRefinement` JSON that nothing in xrd-tools could import.

## Decision

**LAYER, don't force-merge: one canonical declarative `Diffractometer` that holds the
complete instrument and emits BOTH views on demand; a separate `DetectorCalibration` for
static detector calibration; UB stays separate.** A pyFAI 3-rotation model is not, in
general, derivable from an arbitrary circle stack (a 6-circle has more sample axes than
pyFAI represents), so the two derived forms are kept as *adapters*, authored together by a
preset (the single correspondence point) rather than merged into one flat struct.

### The object (`core/geometry/diffractometer.py`)

- **`Diffractometer`** (frozen dataclass) carries the pyFAI half (`rot1/2/3`,
  `incident_angle` `AngleMapping`s), the xu half (`sample_circles`, `detector_circles`,
  `r_i`, `camera`, `hxrd_*`, `circle_motors`), the motor lists, the three xu kwargs dicts,
  and an optional `calibration`. Two pure adapters:
  - `to_pyfai_per_frame(motors)` — byte-equal to `DiffractometerGeometry.derive_per_frame`.
  - `to_qconversion()` / `to_hxrd(energy)` — byte-equal to `DiffractometerConfig.make_hxrd`
    (energy-free `QConversion`; `to_hxrd` wraps it with `en=`). xrayutilities stays a lazy
    import inside the adapters.
  - Presets `two_circle` / `fourc` / `psic` / `sixc` (=psic) / `psic_halpha` author BOTH
    halves. `AngleMapping.sign` doubles as a **fitted scale** (`rot = deg2rad(sign·motor +
    offset)`), so a calibrated goniometer maps on with no new field.
- **`DetectorCalibration`** = `PONI` + `detector_config` (the `{"orientation": N}` mount a
  bare `PONI` drops) + `ImageOrientation` (the raw-array rot/flip/transpose mount). The
  90° panel mount is held once as the pyFAI `detector_config` and once as the xu `camera`
  tuple — both populated by the preset/fit, *not* derived from each other (the
  correspondence is fit-determined; pick by control-point RMS, not by guessing).
- **`Diffractometer.from_pyfai_goniometer(json)`** parses a standard pyFAI
  `GeometryTransformation` (`rotN_expr` strings) into `AngleMapping`s + a base
  `DetectorCalibration`: a *constant* `rotN` → the base `PONI.rotN` (inactive mapping); a
  *pos-linear* `rotN` → an `AngleMapping(sign=scale/deg2rad, offset=offset_rad/deg2rad)`
  with base 0. So per-frame `rotN = calibration.poni.rotN + to_pyfai_per_frame()[rotN]`
  reproduces pyFAI `get_ai(pos)`. Expressions are evaluated with `numexpr`
  (arithmetic-only, no `eval` ACE), linearity is asserted, custom subclasses /
  `ExtendedTransformation` are rejected loud.

### Persistence (ADR-0002 capability-gated, additive)

A capability-gated `/entry/diffractometer` group stores the whole object as one
`config_json` blob (both views + the fitted `DetectorCalibration` + preset + motor map).
`get_diffractometer()` / `ProcessedScan.diffractometer` return `None` on every file
written before the group existed (back-compat, never synthesize a default). **No
`PROCESSED_SCHEMA_VERSION` bump** — it is presence-detected at v2.

### Compat bridges, not a rewrite (step 2)

The two legacy classes stay **authoritative and untouched** for one release.
`Diffractometer.from_diffractometer_geometry` / `to_diffractometer_geometry` and the
`*_config` pair are value-preserving lift/lower bridges, so step 4 can repoint RSM /
reduction / the writer / `Scan.geometry` onto the one object with no behaviour change.

## Rationale

- **The preset is the single correspondence point** — `psic()` authors the xu circle
  stack and the pyFAI `rot1←nu / rot2←del / incidence←eta` mapping together, so the two
  views can no longer disagree. That is the drift fix.
- **Description, not solver.** The object never solves angle→Q; that stays in
  xrayutilities, lazily imported, so `core` stays dependency-light and the `[rsm]` extra
  stays optional.
- **Calibrate once → use for stitch *and* RSM.** The same fitted `Diffractometer` feeds
  the pyFAI per-frame adapter (stitch `MultiGeometry`, the pyFAI q-provider) and the xu
  adapter (`to_qconversion`, the RSM q-provider) — the concrete payoff of one object.
- **Persist it so offline stitch/RSM run from the file with no GUI** — "metadata mandatory
  for stitch/RSM" made concrete, on the existing capability mechanism (no format break).

## Alternatives considered

- **Flat field-merge of the two classes.** Rejected — a pyFAI 3-rotation model is not
  derivable from an arbitrary circle stack; merging would either lose information one side
  needs or smuggle a solver into the dataclass.
- **Auto-derive the pyFAI rotations from the xu circle stack.** Rejected for now — needs
  the per-geometry correspondence rules encoded as data (a bigger lift); preset-authored
  "store both" with a consistency test is simpler and matches today's two-encoding reality.
- **Persist a partial blob from today's `Scan.geometry`.** Rejected — a bare
  `DiffractometerGeometry` has only the pyFAI half (no xu axes / calibration); persisting
  it with default xu axes would mislead an offline RSM consumer. The finish-site write is
  therefore gated on a *complete* `Diffractometer` being present (a no-op until step 4).

## Consequences

- **Implemented + green (headless, steps 0–3):** the `Diffractometer` object, both
  adapters, the presets, `DetectorCalibration` / `ImageOrientation`,
  `from_pyfai_goniometer`, the legacy bridges, and the capability-gated persistence.
  Real-data gate: the goniometer bridge reproduces pyFAI `Goniometer.get_ai` per frame to
  1e-12 on the vendored `gonio_robl_v1/v2.json`. The two legacy classes are untouched, so
  RSM / writer tests pass unchanged.
- **Not yet done (next session — the invasive migration):**
  - **Step 4 — repoint consumers.** RSM `gridding`/`pipeline` → `diff.to_qconversion()`;
    reduction/GI + `write_per_frame_geometry` → `diff.to_pyfai_per_frame()`;
    `Scan.geometry` → a `Diffractometer`; `core/config.diff_config`; the xdart producers
    (`LiveScan.default_geometry` / `_load_from_nexus_v2`). **GI guardrail:** the three
    GI-defining quantities (per-frame `incident_angle`, `gi.tilt_angle`,
    `gi.sample_orientation`) must stay byte-identical; the FiberIntegrator is still built
    from the first frame's incidence and the genuine per-frame incidence applied at
    integration time. This flips the finish-site blob write on with the complete object.
  - **Step 4b — `refine_goniometer`** (the Refine-button backend): a headless
    control-point `least_squares` fit (NOT pyFAI `refine3`, which diverges for the stacked
    psic), seeded from a base `.poni` + calibration images + their `(del,nu)` + calibrant.
    Ships with the stitch/RSM feature, not the core refactor.
  - **Step 5 — retire** the two legacy classes (or leave deprecation aliases) once all
    callers move; grep finds no independent authoring of the two encodings.
- **`circle_motors`** is carried for completeness / persistence but is not yet
  adapter-consumed (RSM passes its motor list explicitly); the psic sample-circle↔motor
  order is the conventional SPEC order, to be cross-validated against real `Ang2Q.area`
  usage when the stitch/RSM gate lands.
- **Version floor (maintainer).** When the consumer rewire (step 4) lands and the writer
  begins emitting/consuming the `diffractometer` group, the `xrd_tools`/release floor must
  cover it — the persistence is additive and back-compat (absent → None), so a new-writer /
  old-reader file simply lacks the group, but a stitch/RSM consumer that *requires* the
  group must gate on the capability.

## Residual risk (honest)

1. **`fourc` xu axes are an unvalidated convention.** No in-repo fixture validates the
   four-circle axis signs; only the structural preset-consistency test guards it, and no
   consumer uses `fourc` yet. Contained — but do not trust its `to_qconversion` until a
   real four-circle calibration validates it.
2. **The mount correspondence is fit-determined, not derived.** `camera` (xu) and
   `detector_config.orientation` (pyFAI) describe the same 90° mount but are stored
   independently; `from_pyfai_goniometer` populates only the pyFAI side (the JSON has no xu
   info), so the xu `camera` must be donated from a `base` preset. A `base` whose convention
   does not match the gonio's real mount would be wrong — the caller is responsible for the
   match.
3. **Step 4 is the dominant regression risk** (the invasive consumer rewire touching the
   writer + GI path). It is deliberately deferred to its own pass with the GI guardrail and
   the live/batch/reload equivalence spine as gates.
