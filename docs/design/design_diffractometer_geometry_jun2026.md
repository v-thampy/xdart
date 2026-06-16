# Design: one shared `Diffractometer` geometry object

**Status:** draft for discussion · 2026-06-14 · planning only (no code)
**Promote to:** ADR-0007 once ratified (this is a "kill parallel representations"
decision that freezes a public API + a persisted group, so it deserves an ADR).
**Consumed by:** [`design_stitching_jun2026.md`](design_stitching_jun2026.md) and the
RSM viewer/reconcile work; both reference this object as their single geometry input.
**Gated on:** nothing in this doc is blocked by 3e/Phase-5 (geometry is orthogonal to
the FrameRecord store collapse), but the *consumer rewiring* should land **after**
3e+Phase-5 is done + tested so it is not a moving target against the store work.
This is its own pervasive `core/geometry` refactor — **do not run it concurrently
with the store collapse** (memory `planned_features_roi_and_stitching_jun2026`).

---

## 1. North-star anchor

- **Kill parallel representations.** Today two objects encode the *same physics* in
  two consumer-specific conventions, authored independently → they can silently
  drift. One canonical declarative description; everything else is a *derived view*.
- **Description, not solver.** The object describes the goniometer; it never solves
  angle→Q. That stays in `xrayutilities`. The dataclass stays pure/light so the
  `[rsm]` extra (xrayutilities) remains a lazy import.
- **Type as data, not a closed enum.** `two_circle` / `fourc` / `sixc` / `psic` /
  `psic_halpha` are *classmethod presets* over a circle stack — a psic and a sixc are
  the same circles in a different order, not different classes.
- **Convention (static, per-instrument) is separate from per-frame angle VALUES.**
  The object carries the circle stack + role→motor-name map; the per-frame motor
  angles live in `scan_data` and are looked up by name at derive time.
- **Persist it, capability-gated.** A reloaded scan must be able to run RSM/stitch
  offline — this is "metadata mandatory for stitch/RSM" (memory
  `stitching_design_reframed`) made concrete.

---

## 2. What exists today (read in full before acting)

### 2.1 `src/xrd_tools/core/geometry/diffractometer.py`
Two **unrelated** containers (the module docstring says so explicitly):

- **`DiffractometerConfig`** — the *xrayutilities / RSM* encoding. Fields:
  `sample_rot`/`detector_rot` (ordered circle stack as axis-direction tuples, e.g.
  `("x+","z-","y+","z-")`), `r_i`, plus HXRD params (`hxrd_n`, `hxrd_q`,
  `hxrd_geometry`), camera orientation (`init_area_detrot`, `init_area_tiltazimuth`),
  and `q_conv_kwargs`/`hxrd_kwargs`/`ang2q_kwargs`. `make_hxrd(energy)` builds
  `xu.QConversion` + `xu.HXRD` (xu imported lazily inside the method).
- **`DiffractometerGeometry`** — the *pyFAI / integration* encoding. Fields:
  `convention: Literal["two_circle","psic","psic_halpha","custom"]`, four
  `AngleMapping`s (`rot1`/`rot2`/`rot3`/`incident_angle`), and
  `sample_motors`/`detector_motors`. Presets `two_circle()`, `psic()`,
  `psic_halpha()`. `derive_per_frame(motors) -> {rot1,rot2,rot3 (rad), incident_angle
  (deg)}`; `all_referenced_motors()`; `to_json`/`from_json`.
- **`AngleMapping`** — scalar `derived = sign * motor + offset` from one motor column;
  `is_active`, `apply()`. (Memory `api_polish_ideas`: *AngleMapping stays scalar* — keep.)

### 2.2 `src/xrd_tools/core/geometry/pixel_q.py`
- **`DetectorHeader`** — `cch1/cch2`, `pwidth1/pwidth2`, `distance`, `Nch1/Nch2`;
  `from_poni(...)` bridges a pyFAI PONI → xu header; `with_roi`, `with_image_shape`.
- **`PixelQMap(diff_config, header)`** — `pixel_q(angles, energy, UB=, roi=,
  image_shape=) -> (qx,qy,qz)`. Bundles `DiffractometerConfig` + `DetectorHeader`.

### 2.3 Wired consumers (the surface a refactor must keep green)
- RSM `rsm/gridding.py` + `rsm/pipeline.py` build `xu.QConversion`/`xu.HXRD` from
  `DiffractometerConfig`; `pixel_q.py` → `PixelQMap`.
- Reduction/GI + stitch use `DiffractometerGeometry.derive_per_frame` for per-frame
  pyFAI rotations + incidence.
- `core.scan.Scan.geometry` holds a `DiffractometerGeometry`; `NexusSink` derives
  `/entry/per_frame_geometry` from `scan_data` at finish.
- `io/schema.py` persists derived `per_frame_geometry` (`rot1/rot2/rot3/
  incident_angle`, capability `per_frame_geometry`). `core/config.py` carries a
  `diff_config`.

**Conclusion from reading the module:** `DiffractometerConfig` (convention as an
xu axis-direction stack) and `DiffractometerGeometry` (motor→pyFAI-rotation map +
per-frame derivation) are **two consumer-specific encodings of one physical
goniometer**, with **no shared fields** and **independently authored presets**. The
`two_circle`/`psic` correspondence between them lives only in a human's head today.

---

## 3. Decision: LAYER (one description + two derived adapters), do NOT force-merge

### 3.1 Why not a single flat merge
A naive "just merge the fields" fails on a real physics mismatch:

- pyFAI's `MultiGeometry`/`AzimuthalIntegrator` models the detector with **exactly
  three rotations** `rot1/rot2/rot3`. A 6-circle goniometer has **more sample axes
  than pyFAI represents**. So you cannot, in general, *derive* the pyFAI rotation
  mapping automatically from the xu circle stack — the mapping is a deliberate,
  per-geometry authored correspondence (which detector circle drives `rot1`, etc.).
- `xrayutilities` needs the full ordered circle stack + axis directions; pyFAI needs
  three rotations + an incidence scalar. These are genuinely different *derived*
  forms of the same instrument.

Forcing one flat struct would either (a) lose information one side needs, or (b)
smuggle a solver into the dataclass. Both violate the north star.

### 3.2 The shape

**One canonical declarative `Diffractometer`** that holds the *complete* instrument
description, and **two pure derived adapter views** computed on demand:

```
                         Diffractometer  (canonical, declarative, dep-light)
                         ├─ preset: str                      # "psic" | "sixc" | ... (DATA)
                         ├─ sample_circles:  tuple[str,...]   # xu axis-direction stack
                         ├─ detector_circles: tuple[str,...]  # xu axis-direction stack
                         ├─ r_i: tuple[float,float,float]
                         ├─ circle_motors: Mapping[circle_index -> AngleMapping]   # xu: which motor drives each circle
                         ├─ rot1/rot2/rot3/incident_angle: AngleMapping            # pyFAI: 3-rot + incidence
                         ├─ sample_motors / detector_motors: tuple[str,...]        # persist verbatim
                         ├─ camera: (init_area_detrot, init_area_tiltazimuth)      # detector-on-arm orientation
                         └─ qconv_kwargs / hxrd_kwargs / ang2q_kwargs
                                   │                                   │
                 ┌─────────────────┘                                   └──────────────────┐
                 ▼                                                                          ▼
   to_pyfai_per_frame(motors)                                              to_qconversion()/to_hxrd(energy)
   → {rot1,rot2,rot3 (rad), incident_angle (deg)}                         → xu.QConversion / xu.HXRD  (lazy xu)
   (today's DiffractometerGeometry.derive_per_frame)                      (today's DiffractometerConfig.make_hxrd)
        consumed by: integration / GI / stitch MultiGeometry                  consumed by: RSM PixelQMap / gridding
```

- **A preset fills BOTH halves consistently.** `Diffractometer.psic(...)` authors the
  xu circle stack *and* the pyFAI `rot1←nu / rot2←del / incidence←eta` mapping in one
  place — the single correspondence point, so the two views can no longer disagree.
  This is the drift fix.
- **`AngleMapping` is reused unchanged** (scalar), composed by the object — both for
  the `circle_motors` (xu side) and the `rot*`/`incident_angle` roles (pyFAI side).
- **Division of labor (pin, write into the ADR):**
  - pyFAI **PONI** = static detector *calibration* (dist, poni1/2, rot1/2/3 of the
    calibration, wavelength, detector). NOT goniometer state.
  - **`Diffractometer`** = goniometer circle stack + per-frame motor angles → sample/
    detector pose → Q (via xu + UB).
  - **UB** = crystal orientation. A *separate* field/persisted group, never inside
    `Diffractometer` (it changes per sample/alignment, not per instrument).
- **Detector camera orientation** (`init_area_detrot`/`tiltazimuth`) belongs to the
  diffractometer convention (it already lives on `DiffractometerConfig`, not
  `DetectorHeader` — keep that boundary). `DetectorHeader` stays the per-detector
  size/pixel/distance carrier feeding `PixelQMap`.

### 3.3 Migration (keep every consumer green)
1. Introduce `Diffractometer` in `core/geometry/diffractometer.py` with the two
   adapter methods and the presets.
2. Re-express `DiffractometerConfig` and `DiffractometerGeometry` as **thin derived
   views / compat shims** over `Diffractometer` (or as `@property`/`classmethod`
   adapters), keeping their current public names + `to_json`/`from_json` for one
   release so RSM pipeline + the writer keep importing what they import today.
3. Repoint consumers to read the one object: RSM `gridding`/`pipeline` call
   `diff.to_qconversion()`; reduction/GI/stitch call `diff.to_pyfai_per_frame(...)`;
   `Scan.geometry` becomes a `Diffractometer`; `NexusSink` derives
   `per_frame_geometry` from `diff.to_pyfai_per_frame(scan_data)`.
4. Retire the two old classes (or leave deprecation aliases) once all callers move —
   the recurring "grep for the parallel representation → nothing" close-out.

---

## 4. Persistence (schema-as-code, additive, capability-gated)

- Add a `diffractometer` (or `/entry/instrument/diffractometer`) group registered in
  `io/schema.py` as a `GroupSchema` + a `CapabilityAttr` (presence-detected, ADR-0002),
  storing the canonical object's `to_json()` blob + `convention`/`preset` + the
  motor-source map. This reconciles the *intended* `/entry/instrument/.../diff_config`
  + UB capture from memory `xdart_rsm_wrangler_extensions`.
- **UB** gets its own capability-gated dataset/group (the reader already returns
  `ub_matrix` in `get_metadata`; formalize the write path under schema-as-code).
- `per_frame_geometry` (derived `rot*`/`incident_angle`) stays exactly as-is — it is
  a derived *cache* of `to_pyfai_per_frame`, re-derivable from the declarative blob +
  `scan_data`. Never rename existing keys (frozen format).
- A reloaded `ProcessedScan` exposes the persisted `Diffractometer` (new
  `metadata["diffractometer"]` / a `scan.diffractometer` property), so offline RSM +
  stitch run from the file with no GUI — the concrete form of "metadata mandatory for
  stitch/RSM."

---

## 5. Open questions — resolved or flagged

1. **Can pyFAI rotations be auto-derived from the xu circle stack?** — **Resolved:
   no, not in general** (pyFAI 3-rotation model ≠ arbitrary goniometer). The preset
   authors both halves. *Flagged as a test:* for each shipped preset, a unit test
   asserts the two derived views are mutually consistent on a synthetic frame
   (catches a mis-authored preset — the very drift this design removes).
2. **Where does the canonical object live?** — `core/geometry/diffractometer.py`,
   replacing the two classes (shims during transition). Keeps the per-frame/per-pixel
   split (`pixel_q.py` still consumes the xu view).
3. **UB ownership.** — Separate field + separate persisted group; never a
   `Diffractometer` field.
4. **`circle_motors` vs the `rot*` mapping — store both or derive one?** — **Flagged
   for Vivek.** Storing both (preset keeps them in sync) is simplest and matches
   today's two-encoding reality; deriving the pyFAI 3-rot from the circle stack is the
   ambitious version but needs the per-geometry correspondence rules encoded as data
   (a bigger lift). Recommendation: **store both, preset-authored**, with the
   consistency test as the guardrail. Revisit auto-derivation only if a third geometry
   makes hand-authoring painful.
5. **Naming.** `Diffractometer` (drop `Config`/`Geometry` suffixes; they become the
   adapter view names if kept at all).

---

## 6. Gated step sequence (each step independently testable; gates front-loaded)

> Land **after** 3e+Phase-5 is done + tested; run as its own branch, not concurrent
> with the store collapse.

0. **Type + presets.** Add `Diffractometer` with `sample_circles`/`detector_circles`/
   `circle_motors`/`rot*`/`incident_angle`/`camera`, the presets (`two_circle`,
   `fourc`, `sixc`/`psic`, `psic_halpha`), and `to_json`/`from_json`. No consumer
   change yet. **Gate:** round-trip JSON; preset-consistency test (§5.1) — both views
   agree on synthetic motors for every preset.
1. **Adapters.** `to_pyfai_per_frame(motors)` (== old `derive_per_frame`, byte-equal
   output) and `to_qconversion()`/`to_hxrd(energy)` (== old `make_hxrd`). **Gate:**
   numeric equality vs the two old classes on captured inputs (the refactor is
   value-preserving by construction).
2. **Compat shims.** `DiffractometerConfig`/`DiffractometerGeometry` become views over
   `Diffractometer` (names + `to_json` preserved). **Gate:** existing RSM + writer
   tests pass untouched.
3. **Persist.** Register the `diffractometer` group + capability in `io/schema.py`;
   write it through the sink; expose `scan.diffractometer` on the reader. **Gate:**
   write→read round-trip; capability feature-detect; back-compat (old files lack the
   group → `None`, no crash); `per_frame_geometry` re-derives identically from the
   blob.
4. **Repoint consumers** (RSM gridding/pipeline, reduction/GI, stitch, `Scan.geometry`,
   `NexusSink`, `core/config.diff_config`) to the one object. **Gate:** full suite +
   the RSM synthetic test; GI matrix unchanged.
5. **Retire** old classes (or leave aliases). **Gate:** grep finds no independent
   authoring of the two encodings.

---

## 7. References
- Code: `core/geometry/diffractometer.py`, `core/geometry/pixel_q.py`,
  `core/scan.py` (`Scan.geometry`, `FrameGeometry`), `io/schema.py`
  (`GroupSchema`/`CapabilityAttr`/`per_frame_geometry`), `rsm/pipeline.py`,
  `rsm/gridding.py`, `io/read.py` (`get_metadata["ub_matrix"]`).
- Decisions: ADR-0002 (capability attrs — the persistence mechanism).
- Memory: `planned_features_roi_and_stitching_jun2026` (the consolidation brief),
  `xdart_rsm_wrangler_extensions` (diff_config + UB capture), `rsm_design_decisions`,
  `stitching_design_reframed`, `api_polish_ideas` (AngleMapping stays scalar),
  `keep_xdart_thin`.
