# Design: intensity corrections (porting pyFAI's + xu grazing-incidence)

**Status:** draft for discussion · 2026-06-20 · planning only (no code)
**Why now:** the xu cake/RSM χ-bug surfaced that `I(q, χ)` and RSM voxel intensities are
load-bearing, not cosmetic (`design_diffractometer_geometry_jun2026.md` §3.5). The *value* in
each (q, χ) / HKL cell is only meaningful once per-pixel intensity corrections are applied. Our
powder-stitch path (pyFAI `MultiGeometry`) already applies some; the **xu RSM/cake path applies
none**, and **grazing-incidence corrections are missing everywhere**.
**Relationship:** corrections are a per-pixel weight stack at the *provider/accumulator* seam —
orthogonal to geometry (`Diffractometer`) and to the stitch/RSM plans. Headless, source-agnostic,
mode-configured.

> **Update 2026-06-23 — the correction stack is the shared pre-weight; the gridder is shared.**
> Stitching has THREE backends (`design_stitching_jun2026.md` §2.6): **`"multigeometry"`** (pyFAI
> MG, 1D+2D), **`"pyfai_hist"`** and **`"xu_hist"`** (pyFAI- vs xu-geometry q-provider → a shared
> per-pixel **histogram merge**). The per-pixel correction stack in this doc is the **shared
> pre-weight** for ALL of them + RSM, applied identically as weights at the provider/accumulator
> seam. The two histogram stitch backends + RSM share the **same** `rsm.gridding.StreamingGridder`
> accumulator (only the bin space differs: (q, χ) for stitch, (qx, qy, qz) for RSM) — so they
> stream. That makes "reuse pyFAI's correction arrays on the xu/histogram path" (§2/§5)
> load-bearing for stitch too, not just RSM. Validated in the `Multi120_*` notebooks (§6).

---

## 1. The canonical correction set (and who applies it today)

The reference list for area-detector GI/powder work (GIXSGUI, Jiang 2015; and the 2024 static-
area-detector treatment) is: **detector flat-field & efficiency, dark current, air-path
absorption, polarization, Lorentz, solid-angle**, plus the **geometric q-reshaping** (which we
already do via pyFAI/xu q-maps). For grazing incidence add **footprint / illuminated-area,
refraction, penetration-depth / absorption, and Fresnel (Vineyard/DWBA) transmission**.

| correction | physics | powder stitch (pyFAI MG) | xu RSM / cake | notes |
|---|---|---|---|---|
| **solid angle** | flat-panel pixels subtend `∝ cos³(2θ)/dist²` | **on by default** (`correctSolidAngle=True`) | **missing** | geometric; identical formula either engine |
| **polarization** | synchrotron is ~linearly polarized; `I ∝ 1−(polarization terms)` | **off unless `polarization_factor` set** | **missing** | must pass `polarization_factor` even for stitch |
| **Lorentz** | powder/scan geometric weighting | not applied (powder MG) | not applied | usually only for powder *integration*, optional for 2D |
| **dark / flat / efficiency** | detector response | `dark=`/`flat=` args (off by default) | **missing** | also `xu.normalize.IntensityNormalizer` (`darkfield`,`flatfield`) |
| **air / sensor absorption, parallax** | beam attenuation in air + sensor depth | `absorption=`, `_correct_parallax` | **missing** | detector/path-level |
| **monitor / count-time** | I0 normalization | `normalization_factor` (scalar) | `xu.normalize` (`mon`,`time`) | per-frame scalar; already a wrangler input |
| **footprint** (GI) | illuminated area `∝ 1/sin αi` | — | — | constant if αi fixed; matters when αi varies |
| **refraction** (GI) | Snell with `n=1−δ+iβ` shifts αi/αf → qz near αc | — | — | distorts qz; build from `Material.idx_refraction` |
| **penetration / absorption** (GI) | evanescent below αc; finite probe depth | — | — | from `absorption_length` + `critical_angle` |
| **Fresnel / Vineyard** (GI) | `|T(αi)|²|T(αf)|²` Yoneda enhancement near αc | — | — | from `chi0` / `simpack` reflectivity |

**Takeaway:** for *powder stitch* the only real gap is turning on **polarization** (solid angle
is already on). For the **xu RSM/cake path, all of solid-angle + polarization (+ optional
dark/flat) are missing**. **GI corrections are missing in every path** and are the bigger
scientific lift.

---

## 2. What xrayutilities actually provides (verified, v1.7.12)

- **`xu.experiment.GID` / `GISAXS`** — GI geometry classes (`Ang2Q`, `Ang2HKL`, `Q2Ang`,
  `TiltAngle`) for the four-circle (αi, azimuth, 2θ, β) surface geometry. **Pure kinematic
  angle→q — refraction is NOT baked in** (verified: no refraction/correction refs in `Ang2Q`).
  Gives the correct *surface-referenced* q; corrections are a separate step.
- **`xu.materials.Material`** — the GI physics primitives, all energy-dependent:
  `idx_refraction(en) → n = 1−δ+iβ`, `delta`, `ibeta`, `critical_angle(en) → αc`,
  `absorption_length(en)`, `chi0`. (Sanity: Si @ 10 keV → αc ≈ 0.179°, μ⁻¹ ≈ 130 µm,
  δ ≈ 4.9e-6, β ≈ 7.6e-8.) These are exactly the inputs for footprint/refraction/absorption/
  Fresnel.
- **`xu.simpack`** — `SpecularReflectivityModel`, `DynamicalModel`, `DiffuseReflectivityModel`,
  Parratt/DWBA machinery. Reusable to compute the **Fresnel transmission / Vineyard factor**
  (it already knows the layer optics); overkill if we only need `|T|²` at (αi, αf).
- **`xu.normalize.IntensityNormalizer`** — monitor (`mon`/`av_mon`/`smoothmon`), count-`time`,
  `darkfield`, `flatfield`, and an `absfun` absorber hook. Covers dark/flat/monitor; **no
  solid-angle or polarization**.
- **NOT provided by xu:** ready-made **solid-angle, polarization, footprint, Lorentz** per-pixel
  correction functions (verified by source grep — the only `lorentz`/`polariz` hits are peak-
  shape `Lorentz1d` and dynamical-diffraction `get_polarizations`, neither a beam correction).

**So:** xu gives us the GI *physics* and the GI *geometry*; it does **not** give the per-pixel
detector/beam correction factors. Those we port from pyFAI's formulas (or read pyFAI's arrays
directly, since we already build pyFAI AIs for the stitch path).

---

## 3. Design: one per-pixel correction stack at the accumulator seam

Both stitch and RSM reduce to "for each pixel: `(q-vector, intensity·weight)` → bin." Today
`weight` is `1` (or a monitor scalar). Introduce a **`corrections` layer** that produces a
per-pixel multiplicative array `C(pixel; geometry, energy, material, mode)` so the accumulated
quantity is `I_raw · C`:

```
C = C_solidangle · C_polarization · C_absorption_air · C_flat⁻¹ · (GI: C_footprint · C_fresnel⁻¹ · C_absorption_film)
```

- **Headless, source-agnostic.** Lives in the core next to the q-map (it needs the same
  per-pixel scattering angles the q-map already computes). Source adapters supply only scalars
  (monitor, αi); the per-pixel physics is shared.
- **Reuse, don't reinvent, the detector/beam factors.** We already construct pyFAI
  `AzimuthalIntegrator`s per frame for the stitch path — `ai.solidAngleArray()` and
  `ai.polarization(factor=…)` give validated arrays for free; for the xu/RSM path either call
  the same pyFAI helpers on the equivalent geometry or port the closed-form factors (both are
  one-liners in the pixel scattering angle). **Powder stitch only needs `polarization_factor`
  switched on.**
- **GI factors from xu primitives.** `footprint = sin(αi)` normalization (only when αi varies);
  `refraction`: correct (αi, αf) via Snell with `n` from `idx_refraction`, recompute qz;
  `absorption/penetration`: from `absorption_length` + `critical_angle`; `Fresnel/Vineyard`:
  `|T(αi)|²|T(αf)|²` from `chi0` (or `simpack` Parratt). Gate behind GI mode + a material/energy.
- **Mode-configured (ties to the wrangler doc).** Plain Int / powder stitch: solid-angle +
  polarization (+ optional dark/flat). GI stitch/RSM: add the GI stack, which needs a **material
  + incidence angle** — new inputs the wrangler must collect in GI mode
  (`design_wrangler_organization_jun2026.md` §3.6/§4).
- **Provenance.** Persist which corrections were applied (+ params: polarization factor,
  material, αi) as capability-gated metadata, so a reloaded scan is self-describing and a
  re-reduction is reproducible.

---

## 4. Implementation plan (gated, additive)

0. **Correction primitives (headless).** `corrections.py`: `solid_angle(geom)`,
   `polarization(geom, factor)`, `air_absorption(geom, μ_air)`, returning per-pixel arrays on
   the existing pixel grid. **Gate:** numeric match to pyFAI `ai.solidAngleArray()` /
   `ai.polarization()` on a shared geometry.
1. **Wire into the accumulator weight.** Multiply into the provider/`StreamingGridder.add`
   weight (and expose on the stitch path). **Gate:** powder stitch with polarization on
   reproduces pyFAI `MultiGeometry.integrate1d(polarization_factor=…)` within tolerance; RSM
   voxel intensities change only by the expected smooth factor.
2. **GI corrections (headless, GI-mode).** `gi_footprint`, `gi_refraction` (Snell via
   `idx_refraction`), `gi_absorption` (`absorption_length`+`critical_angle`), `gi_fresnel`
   (`chi0`/`simpack`). **Gate:** refraction shifts qz in the expected direction and vanishes far
   above αc; Fresnel peaks at αc (Yoneda); footprint ∝ 1/sin αi. Validate against a GIXSGUI/
   literature worked example.
3. **Wrangler + persistence.** GI mode collects material + αi; persist applied-corrections
   provenance. **Gate:** reload → corrections metadata round-trips; re-reduction reproducible.

---

## 5. Open questions
1. **Reuse pyFAI arrays or port the formulas for the xu/RSM path?** Reuse is faster to land and
   already validated; porting drops the pyFAI dependency from the RSM path (it's an `[rsm]`
   extra today). Recommend: reuse pyFAI helpers now, port later if we want RSM pyFAI-free.
2. **Lorentz for 2D/RSM?** Usually a powder-1D thing; flag whether any 2D consumer wants it.
3. **Fresnel/Vineyard depth — `chi0` closed form vs full `simpack` Parratt?** Closed-form `|T|²`
   is enough for single-interface; `simpack` only if layered films matter.
4. **Where does αi come from per frame?** Fixed incidence (scalar) vs an αi-resolved scan
   (per-frame) — the footprint/refraction factors are per-frame in the latter (source-adapter
   metadata, like the detector angles).

---

## 6. References
- Corrections reference: Jiang, *GIXSGUI*, J. Appl. Cryst. 48, 917 (2015) — flat-field/
  efficiency, air absorption, polarization, Lorentz, solid-angle; APS GIXSGUI docs. Recent:
  "Intensity corrections for grazing-incidence X-ray diffraction of thin films using static area
  detectors" (J. Appl. Cryst. 2024) — directly our static-area-detector case.
- xu (v1.7.12, verified): `experiment.GID/GISAXS` (kinematic GI geometry), `materials.Material`
  (`idx_refraction`/`delta`/`ibeta`/`critical_angle`/`absorption_length`/`chi0`), `simpack`
  (`SpecularReflectivityModel`/`DynamicalModel`), `normalize.IntensityNormalizer`
  (`mon`/`time`/`darkfield`/`flatfield`/`absfun`).
- pyFAI: `integrate1d` (`correctSolidAngle`, `polarization_factor`, `dark`, `flat`,
  `absorption`, `normalization_factor`); `Geometry.solidAngleArray`/`polarization`/
  `_correct_parallax`.
- Companions: `design_diffractometer_geometry_jun2026.md` §3.5 (azimuth/2D load-bearing),
  `design_wrangler_organization_jun2026.md` (GI-mode inputs), `design_stitching_jun2026.md` §2.6
  (the three stitch backends the correction stack feeds), `design_rsm_jun2026.md` §3.3 (RSM
  consumes the same stack at the gridder weight).
- Reference notebooks (NOT in-repo — `~/repos/example_notebooks/Stitching/`):
  `Multi120_Diagnose_xu_pyFAI_intensity_discrepancy.ipynb` (the per-pixel correction diff —
  solid-angle/polarization reweight intensity, not ring position),
  `Multi120_GI_Corrections_Explorer.ipynb` (the GI correction stack: footprint / refraction /
  penetration / Fresnel), `Multi120_Compare_xu_vs_pyFAI_del_only.ipynb` (the dual-backend
  head-to-head). RSM fixtures: `~/repos/example_notebooks/RSM/` (`RSM_process.ipynb`).
