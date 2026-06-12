> **HISTORICAL:** design note from the two-repo era; review against the current display-layer seams (docs/ARCHITECTURE.md) before acting on it.

# Stitching architecture — design doc

Status: **draft for discussion** (May 2026)
Authors: Vivek + Claude
Related: `nexus_stitch_refactor_plan.md`, `keep_xdart_thin` direction,
`session_may2026_post18_review` memory.

---

## 1. Guiding principle

At the lowest level, pyFAI's `MultiGeometry` cares about exactly two things
per frame: **the image** and **a fully-formed PONI** (dist, poni1, poni2,
rot1, rot2, rot3, wavelength, detector). Everything else — SPEC files,
NeXus/bluesky runs, Tiled, image+sidecar metadata — exists only to
*produce that per-image PONI list*.

So the architecture has a hard seam:

```
                       ┌─────────────────────────────────────────┐
   metadata source ──▶ │  build a PONI for each image             │  (flexible, per-source)
   (SPEC/NeXus/Tiled/  │  (usually: base PONI + detector angles)  │
    image+sidecar)     └───────────────────┬─────────────────────┘
                                            │  (images, ponis)
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │  MultiGeometry → stitched q / q-χ / 2θ-χ │  (fixed, dumb, shared)
                       └─────────────────────────────────────────┘
```

The bottom box never knows where the PONIs came from. The top box is where
all the diffractometer/geometry/source-format complexity lives. This means
the **different-detector-position** case (two images with different
poni1/dist, not just different rotations) is handled for free — the core
just consumes whatever PONIs it's handed.

`ssrl_xrd_tools` owns both boxes (pure, headless, testable). `xdart` owns
only the *association UX*: which images, which source, how scans are
grouped.

---

## 2. ssrl_xrd_tools: the headless layers

### 2.1 Core primitive (new) — `stitch_ponis`

The real contract. A list of images + a matching list of PONI objects.

```python
def stitch_ponis(
    images: Sequence[np.ndarray],
    ponis: Sequence[PONI],
    *,
    mode: str = "1d",              # "1d" | "2d"
    npt_1d: int = 2000,
    npt_rad_2d: int = 1500,
    npt_azim_2d: int = 720,
    unit: str = "q_A^-1",          # "q_A^-1" | "2th_deg" (→ q-χ / 2θ-χ)
    method: str = "BBox",
    radial_range=None, azimuth_range=None,
    mask: np.ndarray | None = None,
    normalization: Sequence[float] | None = None,
) -> IntegrationResult1D | IntegrationResult2D:
    """Build one AzimuthalIntegrator per (image, poni) pair, feed them to
    MultiGeometry, return the stitched pattern.  No motors, no geometry —
    the PONIs are already fully formed (including any per-image poni1/poni2/
    dist differences, not just rotations)."""
```

This is `create_multigeometry_integrators` generalized: instead of "base
PONI + rot1/rot2 offsets" it takes a complete PONI per image. The existing
`stitch_1d` / `stitch_2d` bodies (the actual `mg.integrate1d/2d` calls +
`IntegrationResult` packing) are reused unchanged.

### 2.2 Existing `stitch_images` becomes a thin caller

`stitch_images(images, base_poni, rot1_angles, rot2_angles, …)` stays as a
convenience wrapper for the common "shared base PONI, only rotations vary"
case — it builds the per-image PONI list (copy base, set rot1/rot2/rot3)
and delegates to `stitch_ponis`. No behavior change for existing callers
(xdart `run_stitch`, the headless tests).

### 2.3 PONI-builder layer (new) — `ponis_from_scan`

The bridge from "scan metadata" to "per-image PONIs". This is where the
diffractometer-geometry flexibility lives.

```python
def ponis_from_scan(
    base_poni: PONI,
    motor_table,                  # dict[str, np.ndarray] or DataFrame: motor → per-frame values
    geometry: DiffractometerGeometry,
) -> list[PONI]:
    """One PONI per frame.  Copies base_poni and sets rot1/rot2/rot3 from
    the detector-angle motors via geometry.derive_per_frame().  Sample-axis
    motors (th/phi/chi/eta/mu) are carried for provenance but don't change
    the PONI — only detector angles feed MultiGeometry."""
```

For the rarer "different detector *position*, not just angle" case, a
source can build the PONI list directly (e.g. one base PONI per detector
position) and skip this helper — `stitch_ponis` doesn't care.

### 2.4 Diffractometer geometry presets

`DiffractometerGeometry` already maps motor names → pyFAI rotations
(`derive_per_frame`, `sample_motors`/`detector_motors`,
`all_referenced_motors`). We add **named constructors** so the
detector-angle motors have sane defaults per circle count, with overrides:

| Geometry  | Detector-angle motors (feed MultiGeometry) | Typical sample motors |
|-----------|--------------------------------------------|-----------------------|
| 2-circle  | `tth`                                      | `th`                  |
| 4-circle  | `tth`                                      | `th, chi, phi`        |
| 6-circle  | `del`, `nu`                                | `eta, chi, phi, mu`   |

```python
# Actual constructor names in core/geometry/diffractometer.py:
DiffractometerGeometry.two_circle(tth="tth", th="th", gonchi=None)        # EXISTS
DiffractometerGeometry.psic(del_="del", nu="nu", eta="eta",
                            chi="chi", phi="phi", mu="mu")                 # EXISTS (6-circle)
DiffractometerGeometry.psic_halpha(del_="del", nu="nu", halpha="halpha",
                                   chi="chi", phi="phi", mu="mu")         # EXISTS (6-circle variant)
DiffractometerGeometry.four_circle(tth="tth", th="th", chi="chi", phi="phi")  # new — trivial
```

**Already implemented** (`core/geometry/diffractometer.py`): `two_circle`
(detector `tth`, optional `gonchi` sample motor), `psic` and `psic_halpha`
(6-circle; detector `del`/`nu`, sample `eta|halpha,chi,phi,mu`), plus
`derive_per_frame`, `all_referenced_motors`, `sample_motors`/`detector_motors`,
`AngleMapping` (motor→pyFAI rotation), and the generic dataclass constructor
with explicit `sample_motors=`/`detector_motors=` + per-rotation
`AngleMapping` overrides. So an arbitrary geometry is *already* expressible
today.

**Only genuinely new:** a `four_circle` convenience constructor — and it's
essentially `two_circle` with extra sample motors (`chi`, `phi`), since the
4-circle detector angle is still `tth`. (NB: the 6-circle constructors are
named `psic` / `psic_halpha`, not `six_circle`; a `six_circle` alias could be
added for discoverability but isn't required.)

Only the detector angles affect the stitch; the rest are metadata. Motor
names are overridable because beamlines label them differently. **Net: the
geometry layer needs almost no new code — the presets are essentially done.**

### 2.5 Source enumeration (new) — `StitchSource`

A thin interface so xdart doesn't reimplement scan-listing per format.
Each backend wraps readers that mostly already exist in `xrd_tools.io`.

```python
class StitchSource(Protocol):
    def list_scans(self) -> list[ScanId]: ...
    def load_scan(self, scan_id) -> LoadedScan:
        # → images, motor_table, base_poni (or per-image ponis), metadata
```

| Backend                  | Wraps                                   | Priority |
|--------------------------|-----------------------------------------|----------|
| `SpecStitchSource`       | `io.spec` (get_scan_path_info, get_angles) | **1st** |
| `NexusStitchSource`      | `io.nexus` (bluesky-written)            | **2nd** |
| `TiledStitchSource`      | `io.tiled` (connect_tiled, read_tiled_run) | **3rd** |
| `ImageSeriesStitchSource`| current image+sidecar path, recast      | later   |

`load_scan` returns motor positions + a base PONI; xdart (or a default
helper) calls `ponis_from_scan` to get the per-image PONIs, then
`stitch_ponis`. Result saved with the existing `write_stitched`.

### 2.6 What already exists vs. new

Already done: `stitch_1d`/`stitch_2d`, `stitch_images`, `write_stitched`,
`read_stitched`, the full `DiffractometerGeometry` (`derive_per_frame`,
`two_circle`, `psic`, `psic_halpha`, `AngleMapping`, generic override
constructor, `all_referenced_motors`), the SPEC/NeXus/Tiled readers,
`IntegrationResult1D/2D`.

New ssrl code: `stitch_ponis` (small — generalizes integrator-build),
`ponis_from_scan` (small), optional `four_circle` preset (trivial),
`StitchSource` + backends (the bulk, but each is a thin reader wrapper). The
geometry layer is effectively done.

---

## 3. xdart: the stitch wrangler

### 3.1 Mode dropdown vs. source selector (two axes)

The Mode dropdown gains **Stitch 1D** and **Stitch 2D** (alongside Int 1D,
Int 2D, …). Picking either swaps the wrangler stack to a **stitch
wrangler**. Inside that wrangler, a **source selector** chooses where the
data comes from:

```
Mode:   [ Int 1D | Int 2D | Int 1D (XYE) | Stitch 1D | Stitch 2D | Image Viewer | XYE Viewer ]
                                              └──────────┬──────────┘
                                                         ▼  swaps wrangler stack to:
Source: [ SPEC | Directory | Nexus | Multi ]            (stitch wrangler)
```

Rationale: Mode = *what to produce* (1D vs 2D stitch); Source = *where
from*. Keeping them as separate axes avoids overloading one dropdown with
both concepts, and reuses the existing wrangler-stack swap machinery.

### 3.2 The four sources

| Source        | Front-end fields                                        | Output granularity |
|---------------|--------------------------------------------------------|--------------------|
| **SPEC**      | spec file; scan-number selection with range syntax: `1-10`, `15,16,18`, `30-40`; image-path (defaults to spec dir); diffractometer geometry preset + motor overrides | one stitched pattern per scan (or per group) |
| **Directory** | directory + filename filter; watches for new spec files and stitches each scan as it lands (live) | per scan, streaming |
| **Nexus**     | nexus file(s); mirrors the current Image Series flow (motor datasets → angles) | per scan/run |
| **Multi**     | **see open question** — combine several named scans/files into ONE stitched output | one combined pattern |

Shared stitch params (all sources): mode 1D/2D (from the Mode dropdown),
`norm_motor`, npt, radial/azimuth ranges, mask, geometry preset.

### 3.3 Scan grouping

For SPEC the scan-number field is also the grouping control:
- `1,2,3` → three separate stitched outputs (one per scan).
- `1-3` → the frames of scans 1, 2, 3 stitched into one output.
- `1-3, 5, 7-9` → group {1,2,3}, single {5}, group {7,8,9} → three outputs.

This makes "each scan individually" vs "several scans together" a single
expressive field rather than a separate mode.

---

## 4. End-to-end flow (SPEC example)

1. User: Mode = **Stitch 1D**, Source = **SPEC**, picks `combi.spec`,
   scans `1-3, 5`, geometry = 4-circle (tth detector motor).
2. xdart's stitch wrangler builds a `SpecStitchSource(combi.spec)`.
3. For group `{1,2,3}` and single `{5}`:
   a. `source.load_scan(...)` → images + motor_table + base PONI.
   b. `ponis_from_scan(base_poni, motor_table, geometry)` → per-image PONIs.
   c. `stitch_ponis(images, ponis, mode="1d", …)` → `IntegrationResult1D`.
   d. `write_stitched(entry, stitched_1d=result)` → `.nxs`.
4. Display + save reuse the existing stitched-pattern path.

Steps 3a–3d are all headless ssrl; xdart only does the source construction
and the grouping loop.

---

## 5. Open questions

1. **Multi semantics.** Is "Multi" a *source* (cross-format combine of
   named scans/files into one output) or really a *grouping* mode that
   could apply within any source? If grouping is already expressible via
   SPEC's range syntax, Multi may be redundant — or it's specifically
   "combine across different files/formats", which is a distinct need.
   → decide whether grouping is a per-source option or its own entry.

2. **Per-source motor→angle mapping.** How much is auto-detected
   (SPEC #O/#P header motors; bluesky positioner datasets; tiled metadata)
   vs. user-specified via the geometry preset? Proposal: auto-detect the
   candidate motors, preset picks the defaults, user can override names.

3. **Different detector-position case.** Confirm the expected workflow:
   does the user supply multiple PONI files (one per position), or is it
   derived from a translation motor? `stitch_ponis` supports both; the
   question is the xdart input UX.

4. **Live directory stitching.** Reuse the existing directory-watch
   machinery from the image wrangler? Each new spec file → enumerate its
   scans → stitch per the grouping rule.

5. **Output naming/location.** One `.nxs` per stitched output; naming
   convention for grouped outputs (e.g. `combi_scans_1-3_stitched.nxs`).

---

## 6. Phasing

- **P0** ssrl core: `stitch_ponis` + `ponis_from_scan` + `four_circle`/
  `six_circle` presets + headless tests. (No xdart yet — fully testable.)
- **P1** `SpecStitchSource` + a minimal xdart stitch wrangler (SPEC source
  only, scan-range syntax, Stitch 1D/2D mode). End-to-end on real spec data.
- **P2** `NexusStitchSource` + Nexus source in the wrangler.
- **P3** Directory (live watch) + filter.
- **P4** `TiledStitchSource`; Multi (pending the open question).
