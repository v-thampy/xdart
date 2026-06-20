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
  **Key (Jun 2026, validated):** `sign` doubles as a **fitted dimensionless scale** —
  `derive_per_frame` folds `deg2rad` in, so the effective rot is `sign·deg2rad(motor) +
  deg2rad(offset)`. A *fitted* pyFAI goniometer (`rot = rot_scale·motor + rot_offset`,
  with `rot_scale ≈ 0.96–0.99·deg2rad`) maps onto it exactly: `sign = rot_scale/deg2rad`,
  `offset = rot_offset/deg2rad` (degrees). So **`AngleMapping` already carries a fitted
  calibration with no new field** — see §3.4.

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
  - **`DetectorCalibration`** = static detector *calibration*: dist, poni1/2, rot1/2/3
    of the calibration, wavelength, detector **and its `Detector_config`** (orientation/
    mask/binning). NOT goniometer state. *(Today this is `core/containers.PONI`, which
    drops `Detector_config` — the stitching notebook proved this is a real gap; either
    extend `PONI` with `detector_config: dict` or add a `DetectorCalibration` wrapper.
    See `design_stitching_jun2026.md` §2.5 GAP B / §3.1.)*
  - **`Diffractometer`** = goniometer circle stack + per-frame motor angles → sample/
    detector pose → Q (via xu + UB). Carries the **fitted** per-axis scale+offset
    (§3.4), not a hardwired `deg2rad`.
  - **UB** = crystal orientation. A *separate* field/persisted group, never inside
    `Diffractometer` (it changes per sample/alignment, not per instrument).
- **The detector mount (e.g. a 90° panel rotation) has TWO encodings — unify them.**
  pyFAI expresses it as `Detector_config {"orientation": N}` (consumed by the
  integration/stitch AI); xrayutilities expresses the same mount as the camera
  orientation `init_area_detrot`/`init_area_tiltazimuth` (consumed by RSM). These are
  the *same physics* in two conventions — exactly the parallel representation this object
  kills. **`Diffractometer` holds the mount once and emits both**: the pyFAI
  `Detector_config` (via the `DetectorCalibration`) and the xu camera tuple. The
  stitching notebook found that the 90° must live in the detector, never as `rot3`.
  `DetectorHeader` stays the per-detector size/pixel/distance carrier feeding `PixelQMap`.

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

### 3.4 The fitted pyFAI goniometer IS the production instance (validated Jun 2026)

The stitching notebook (`examples/.../Stitching/stitch_simplified.ipynb`, real LaB6 +
Pilatus 300k-w data) showed the real-world geometry artifact is a pyFAI
`GoniometerRefinement` JSON: a `GeometryTransformation` (`rot1 = rot1_scale·nu +
rot1_offset`, `rot2 = rot2_scale·del + rot2_offset`, `rot3 = rot3_offset≈0`) over a
detector carrying `Detector_config {"orientation": 3}`. That is **exactly an instance of
`Diffractometer`** — fitted, not preset. So the object needs two thin entry points
(both wrap pyFAI; both keep xu lazy):

- **Consume — `Diffractometer.from_pyfai_goniometer(json)`.** Parse the
  `GeometryTransformation` expressions into `AngleMapping`s (scale→`sign`,
  offset→`offset`; §2.1) and the base params into a `DetectorCalibration` (incl.
  `Detector_config`). A beamline calibrates in pyFAI; xrd-tools then stitches/RSMs
  headlessly with **no pyFAI `Goniometer` at runtime**. (The notebook only calls
  `Goniometer.sload` directly because this bridge is missing — it's the single most
  useful interop step.)
- **Produce — `refine_goniometer(base_calibration, images, angles, calibrant, *,
  preset, fit, bounds) -> Diffractometer`.** A headless wrapper (mirrors the `run_*`
  analysis-plan pattern), the backend for xdart's **"Refine" button** (Vivek, Jun 2026).
  **Robust backend = control-point least-squares, not `refine3`** — see §3.5: pyFAI
  `GoniometerRefinement.refine3` (simplex) works for the simple 1-DOF preset but
  *diverges* for the harder stacked psic; `scipy.least_squares` on the control-point
  q-residual is stable for both and is the validated path.

**Calibration workflow (Vivek's idea — matches the Multi120 notebook):**
1. Run pyFAI-calib on **one** low-angle image at a known `(del, nu)` → a single-position
   base **`.poni`** (+ control points). This is the *starting point*, not the answer.
2. Seed `refine_goniometer` with that base `.poni`'s `dist/poni1/poni2/rot1/rot2/rot3`
   as the initial guess.
3. Add calibration images across the `del`/`nu` range; auto-extract rings (`extract_cp`)
   and fit the goniometer offsets **and** scales (which params are fit vs fixed is an
   input).
4. Output: a `Diffractometer` (+ `DetectorCalibration`) — the saved goniometer.
   - **xdart:** the Calibration control gains a **"Refine"** button that takes
     `(base .poni, calibration images + their (del,nu) metadata, calibrant)` → runs
     `refine_goniometer` → stores the `Diffractometer` for stitch/RSM. Thin button;
     refinement is the headless function.

**RSM reuse (Vivek: "can be used for RSM as well").** The *same* fitted `Diffractometer`
+ mount feeds RSM's `to_qconversion()`/`PixelQMap`: the mount → the xu camera tuple, the
per-frame motors → `Ang2Q.area` angles, UB → orientation. **Calibrate once → use for
stitch *and* RSM** — the concrete payoff of the one-object design. (Caveat: RSM's UB +
energy still come from the scan, not the goniometer; only the *instrument* geometry is
shared.)

**Preset-built vs fitted — refinement is optional.** A `Diffractometer` is either
**preset-built** (uncalibrated: scale = `deg2rad`, base from a single `.poni` — the
`AzInt(rot1=nu, rot2=del)` route) or **gonio-fitted** (calibrated, via
`from_pyfai_goniometer`). Both flow through the *same* adapters, so consumers never
branch on it; the only difference is accuracy — a slow del-dependent drift from the
unfitted scale. (The fixed-order `nu·del` cross-term is shared by **both** paths and was
measured negligible on real data — full mesh == narrow band — so neither path needs an
angle guard.) See `design_stitching_jun2026.md` §3.1.

### 3.5 Validated refinement recipe (Jun 2026 — del/nu mesh + del-only cross-check)

Two notebooks now pin down *how* `refine_goniometer` should fit, on real LaB6 + Pilatus
300k-w data (`examples/.../Stitching/Multi120_Calibration_Pilatus300kw_del_nu_SURFACE.ipynb`
for the hard stacked-psic del/nu case, and `Multi120_Compare_xu_vs_pyFAI_del_only.ipynb`
cross-validating against the pyFAI `ROBL_v1.json` from `Multi120_Pilatus300kw.ipynb`).

- **Fit against identified control-point rings with `scipy.least_squares`, not `refine3`.**
  Parse pyFAI `.npt` control points (ring index → LaB6 |q| = 2π/d), and minimise the
  per-point q-residual. This is the *same* method pyFAI's `GoniometerRefinement` uses, but
  the `least_squares` driver is stable where `refine3`'s simplex diverges to non-physical
  minima for custom multi-param transforms (stacked-arm, surface). RMS **0.009 Å⁻¹** on
  the del/nu mesh; **0.005** del-only.
- **Either engine produces the canonical object; they agree.** Fitting the **xu**
  `QConversion` directly (params `cch1, cch2, dist, del_offset, nu_offset`) yields the
  RSM-native view straight away; fitting a **pyFAI** matrix-decomposition stacked-arm
  (true pose `R_ν·R_δ` → decompose to `rot1/rot2/rot3`, cross-term in rot3) yields the
  integration view. Both fit the same control points to RMS 0.009 and land within ~4 %
  peak/median on the full mesh — concrete evidence the two adapter views (§3.2) describe
  one physics. (`refine_goniometer` may fit in whichever engine is convenient and populate
  the canonical `Diffractometer`; the preset authors the other view.)
- **Pick the detector mount by RMS, don't guess.** For xu the camera orientation is the
  `init_area` axis-sign pair (`camera_or`); sweeping the four combos and taking the one
  with the lowest control-point RMS is decisive here (`('x-','z+')` gave 0.009 vs ≥0.29
  for the others). The chosen mount is the *same* physics as the pyFAI `Detector_config`
  orientation (§3.2) — the object stores it once and emits both.
- **Motor-zero offsets are first-class fit parameters and were the missing ingredient.**
  The del/nu fit recovered `del` ≈ **+3.0°**, `nu` ≈ **+1.2°** zero errors — without them
  *no* geometry lands the rings (this is the long-standing "del-only calibrates but del+nu
  doesn't" symptom). They live in `AngleMapping.offset` per role (§2.1), so the canonical
  object already carries them — `refine_goniometer` just has to fit them.
- **Conditioning caveat — the fit is only as good as the angular spread of the control
  points.** The beam-centre channel on the *short* detector axis is weakly constrained: on
  the del/nu mesh (control points only to `nu = 9.5°`) `cch1` drifted to 221 while still
  fitting |q| to RMS 0.009; on del-only (rings sweep that axis) it pinned to **193.2**,
  matching pyFAI's fitted `poni1/pix = 192.6` to **< 1 px**. *Guidance to surface in the
  Refine UI:* control points should span the full range of each fitted axis (esp. add
  high-`nu` frames); report the fitted centre next to the picker's seed so a large drift is
  visible. (The `db_pixel` "direct-beam" value beamlines quote is only a picker seed — pyFAI
  itself refines it ~100 px away.)
- **No engine-specific geometric correction to worry about.** pyFAI's `Pilatus300kw` is a
  uniform 172 µm grid (zero module-gap jumps), identical to xu's flat `init_area`; pyFAI's
  only extras (solid-angle, polarisation) reweight *intensity*, not ring position. On
  del-only the xu and pyFAI 1-D stitches overlay essentially perfectly. So the two views
  stay numerically equivalent **on |q|** — see the next point for the azimuth.
- **The powder |q| fit does NOT constrain the in-plane / azimuthal orientation — and that
  orientation IS load-bearing for 2-D stitch and RSM.** A powder ring is azimuthally uniform,
  so `χ` (equivalently the *direction* of **q**, not just |**q**|) drops out of the 1-D fit and
  the |q| RMS. But the 2-D product — `I(q, χ)` cakes (texture, preferred orientation, GI) and
  the full 3-D RSM voxel placement — depends entirely on it. Two consequences the validation
  must respect: **(a)** compute the cake/RSM azimuth from the components *transverse to the
  beam* (in `HXRD([0,1,0],…)` the beam is along **Y**, so `χ = −atan2(qx, qz)`, **not** a
  `qy`-mixed form — getting this wrong over-spreads χ and aliases intensity into the wrong bins;
  it matched pyFAI `chiArray` to 0.03° once corrected); **(b)** the refinement/gate must
  *validate the azimuthal orientation independently* (cross-check χ against a reference geometry,
  or a known reflection for RSM) rather than declaring success on |q| RMS alone. For RSM this is
  not optional — the q-vector direction is the measurement.

---

## 4. Persistence (schema-as-code, additive, capability-gated)

- Add a `diffractometer` (or `/entry/instrument/diffractometer`) group registered in
  `io/schema.py` as a `GroupSchema` + a `CapabilityAttr` (presence-detected, ADR-0002),
  storing the canonical object's `to_json()` blob + `convention`/`preset` + the
  motor-source map. This reconciles the *intended* `/entry/instrument/.../diff_config`
  + UB capture from memory `xdart_rsm_wrangler_extensions`.
- Persist the **`DetectorCalibration`** alongside it — including `Detector_config`
  (orientation/mask/binning) and wavelength — so a reloaded scan reconstructs the full
  per-frame geometry (mount included) for offline stitch/RSM. A pyFAI goniometer JSON
  must round-trip through `from_pyfai_goniometer` → persist → reload without loss.
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
1b. **`DetectorCalibration` + pyFAI-goniometer bridge.** Add `DetectorCalibration`
   (PONI fields + `detector_config`); `Diffractometer.from_pyfai_goniometer(json)`; emit
   the mount as both pyFAI `Detector_config` and the xu camera tuple. **Gate (real
   data):** load `MG_gonio_object.json` from the stitching notebook → per-frame geometry
   equals pyFAI `Goniometer.get_ai((del,nu))` within tolerance; `Detector_config`
   round-trips; the fitted scales (not `deg2rad`) are recovered.
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
4b. **`refine_goniometer` producer (Refine-button backend).** Headless control-point
   `least_squares` fit (§3.5, **not** `refine3`), seeded from a base `.poni` + calibration
   images + their `(del,nu)` metadata + calibrant; recovers per-axis scale+offset (incl.
   motor-zero offsets) and the detector mount. Ships with the stitching/RSM feature, not the
   core refactor. **Gate (real data):** re-fit over the notebook's LaB6 images reproduces the
   saved goniometer params; del-only fit matches pyFAI `ROBL_v1.json` beam centre to < 1 px
   and stitch sharpness within ~5 %; the recovered `del`/`nu` offsets are non-zero.
5. **Retire** old classes (or leave aliases). **Gate:** grep finds no independent
   authoring of the two encodings.

---

## 7. References
- Code: `core/geometry/diffractometer.py`, `core/geometry/pixel_q.py`,
  `core/scan.py` (`Scan.geometry`, `FrameGeometry`), `io/schema.py`
  (`GroupSchema`/`CapabilityAttr`/`per_frame_geometry`), `rsm/pipeline.py`,
  `rsm/gridding.py`, `io/read.py` (`get_metadata["ub_matrix"]`).
- Decisions: ADR-0002 (capability attrs — the persistence mechanism).
- Validated by: `examples/.../Stitching/stitch_simplified.ipynb` + `MG_gonio_object.json`
  (real pyFAI goniometer: the fitted `rot=scale·motor+offset` model + `Detector_config`
  orientation that this object must represent). Companion: `design_stitching_jun2026.md`
  §2.5 (GAPs A–D) and §3.1–3.2a (the `DetectorCalibration`/`stitch_ponis`/bridge it needs).
- Refinement recipe (§3.5) validated by: `Multi120_Calibration_Pilatus300kw_del_nu_SURFACE.ipynb`
  (stacked-psic del/nu, xu vs pyFAI stacked-arm, control-point `least_squares`, recovered
  `del/nu` offsets) and `Multi120_Compare_xu_vs_pyFAI_del_only.ipynb` (del-only cross-check
  vs pyFAI `ROBL_v1.json`: beam centre < 1 px, equal stitch sharpness, uniform-grid detector).
  Memory: `psic_del_nu_calibration_solved_jun2026`.
- Memory: `planned_features_roi_and_stitching_jun2026` (the consolidation brief),
  `xdart_rsm_wrangler_extensions` (diff_config + UB capture), `rsm_design_decisions`,
  `stitching_design_reframed`, `api_polish_ideas` (AngleMapping stays scalar),
  `keep_xdart_thin`.
