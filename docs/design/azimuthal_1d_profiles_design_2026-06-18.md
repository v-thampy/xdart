# Azimuthal 1D profiles (I vs χ) — design / handoff spec

**Status:** specced. **Mode A (standard / non-GI): targeted for v1.0**
(the "third Int-1D option" beside Q and 2θ). **Mode B (grazing incidence):
deferred post-v1.0.** Author: review pass, 2026-06-18.

Both modes are the same feature — *integrate over a q band, output intensity vs
an angle* — with one shared UX and one shared display path. They differ only in
the pyFAI call. Mode A lands first and builds the angle-axis display that Mode B
reuses.

---

## 0. Summary

Add an **azimuthal-profile** Int-1D mode: pick a q (or q_total) band, get
**I vs angle**:

| | angle axis | pyFAI call | scope |
|---|---|---|---|
| **A. standard (non-GI)** | χ (detector azimuth, `chi_deg`) | `AzimuthalIntegrator.integrate_radial` | **v1.0** |
| **B. grazing incidence** | χ_GI = arctan(q_oop/q_ip) (`chigi_deg`) | `FiberIntegrator.integrate1d_polar(radial_integration=True)` | post-v1.0 |

Both are **native pooled integrations** — the correct, count-weighted quantity —
not cake-row projections (which carry the mean-of-means error, see §1).

---

## 1. Motivation (shared)

Established during the GI-projection review (`review_2026-06-15*` + the
conversation that produced this doc):

1. **Every pyFAI 1D integrator pools** `I = Σ signal / Σ normalization` — it
   accumulates signal and normalization separately across the collapsed axis and
   divides **once** (count/normalization-weighted). This is true for
   `integrate1d`, `integrate1d_fiber`, `integrate_radial`, and
   `integrate1d_polar` alike — it lives in the shared histogram/CSR engine, not
   in any one wrapper.
2. The GUI's **cake→1D projection** instead takes an unweighted `np.nanmean` of
   already-normalized cake cells — a *mean-of-means* that differs from the pooled
   result wherever per-cell pixel count varies along the collapsed axis
   (Simpson's-paradox effect; ~1% median, larger at peaks/edges/sparse cells).
3. An **azimuthal profile** (I vs angle over a q band) is a *different observable*
   from the radial I(q): it carries texture / preferred-orientation information.

Deriving it by collapsing the displayed cake would re-introduce the mean-of-means
error and the missing-wedge contamination. The native integrators do it pooled
and correct — which is why this is its own mode, not a projection.

---

## 2. Mode A — standard (non-GI) χ via `integrate_radial`  [v1.0]

### 2.1 pyFAI call (verified, pyFAI 2025.3.0)
```python
ai.integrate_radial(data, npt,                 # npt = number of χ output bins
                    npt_rad=<q samples>,       # radial integration sampling
                    radial_range=(qmin, qmax), # the q (or 2θ) band to integrate
                    radial_unit="q_A^-1",      # or "2th_deg"
                    unit="chi_deg",            # output axis = χ (degrees)
                    method=("bbox","csr","cython"),
                    mask=mask, correctSolidAngle=True,
                    polarization_factor=..., normalization_factor=...)
# -> result.radial = χ (deg, -180..180), result.intensity = I(χ), result.count present
```
Confirmed: returns a real I(χ) on a `chi_deg` ±180° axis with `.count`; recovers
a known azimuthal modulation; differs from a cake-row projection by ~1% median
(the mean-of-means effect — this is the more precise quantity).

### 2.2 Headless wrapper — `xrd_tools/integrate/single.py`
Add next to `integrate_1d` (single.py:22-99):
```
def integrate_radial(image, ai, npt=1000, radial_unit="q_A^-1", method="csr",
                     mask=None, radial_range=None, error_model=None,
                     polarization_factor=None, normalization_factor=None,
                     **kwargs) -> IntegrationResult1D:
    result = ai.integrate_radial(image, npt, unit="chi_deg",
                                 radial_unit=radial_unit, radial_range=radial_range,
                                 method=method, mask=mask, **extra)
    return IntegrationResult1D(radial=<χ axis>, intensity=<count==0 → NaN>,
                               sigma=..., unit="chi_deg")
```
- NaN-fill empty (count==0) χ bins via the shared helper (see §4) — keep genuine
  zeros (count>0). Transmission usually has near-full azimuth, but gaps/edges
  still produce empties.
- Confirm `IntegrationResult1D` accepts a non-q radial axis + arbitrary unit
  string ("chi_deg"); the container already stores a `unit` string and the 2D
  result already uses `chi_deg` for `azimuthal_unit`, so this should be fine —
  verify no q≥0 / monotonic-q assumption in the container.

### 2.3 GUI touch points — `integrator.py`
- **Units list** (integrator.py:45-48): `Units = [Q, 2θ]`,
  `Units_dict = {Q:'q_A^-1', 2θ:'2th_deg'}`, `Units_dict_inv`. Append a χ entry
  → `'chi_deg'`. The existing `_get_unit_1D`/`_set_unit_1D` (862-874) then handle
  it automatically via the dict.
- **Combo population** (integrator.py:150-151): add the third `setItemText`.
- **Range validators** (integrator.py:431-448): the χ mode's *x-axis* is the angle
  (±180°), but the editable **range field is the q (or 2θ) band to integrate
  over** — reuse the existing radial-range field for that band; do not treat the
  range field as a χ-axis range. Make sure validation matches (q band positive
  for Q; ±180 only if a χ clip is exposed).
- **Pts defaults** (`_NPTS_1D_DEFAULTS`, 1374-1383): the standard `'std'` key
  works; `npt` = χ output bins, with `npt_rad` (q sampling) defaulted in the
  wrapper.

### 2.4 Frame dispatch — `frame.py` (standard branch, ~558-620)
```
if not self.gi:
    if str(unit) == 'chi_deg':
        result = integrate_radial(image, self.integrator, npt=numpoints,
                                  radial_unit=<'q_A^-1' or '2th_deg'>,
                                  radial_range=radial_range, mask=mask,
                                  method=..., **std_kwargs)
    else:
        result = integrate_1d(..., unit=str(unit), radial_range=radial_range, ...)
    self.int_1d = result
```
`radial_range` here is the q/2θ band to integrate over (the existing field).
Decide the band unit: simplest is always Q; or follow a secondary Q/2θ toggle.

---

## 3. Mode B — grazing incidence χ_GI via `integrate1d_polar`  [post-v1.0]

### 3.1 pyFAI call (verified)
`FiberIntegrator.integrate1d_polar` (pyFAI/integrator/fiber.py):
```python
# radial_integration: False -> I vs q_total ; True -> I vs polar angle χ_GI
unit_oop = "chigi_deg" if polar_degrees else "chigi_rad"   # χ_GI = arctan(qOOP/qIP)
unit_ip  = "qtot_A^-1" if radial_unit=="A^-1" else "qtot_nm^-1"
kwargs["vertical_integration"] = bool(radial_integration)
```
The deciding flag is **`radial_integration=True`** → output **I vs χ_GI**
(`chigi_deg`), pooled over q_total. Reference call:
```python
fi.integrate1d_polar(data=img, sample_orientation=so,
                     radial_integration=True, polar_degrees=True, radial_unit="A^-1",
                     npt_oop=500,           # χ_GI output bins
                     npt_ip=1000,           # q_total integration samples
                     ip_range=(q0, q1),     # q_total band to integrate over
                     method="no", mask=mask)
# -> result.radial = χ_GI (deg, ~-180..180), result.intensity = I(χ_GI), count present
```
**Parameter inversion (the `vertical_integration` gotcha):** because
`unit_ip = q_total` and `unit_oop = χ_GI`, the **output χ bins = `npt_oop`**, the
**q sampling = `npt_ip`**, and the **q band = `ip_range`** (an optional χ clip is
`oop_range`). Use `polar_degrees=True` for a ±180° axis (the verified variant
used `False` → radians).

### 3.2 What is already in place
- `xrd_tools.integrate.gid.integrate_gi_polar_1d` (gid.py:573) already wraps
  `integrate1d_polar` for the `q_total` mode and already lists
  `radial_integration`/`polar_degrees` as forwardable kwargs (gid.py:627), but it
  always returns the **q_total** axis (radial_integration left False).
- `integrate_gi_exitangles_1d` (gid.py:696) already exists — so the codebase
  already produces a **non-q GI 1D axis** (exit angles). The display layer
  therefore likely already handles an angle-valued GI 1D x-axis; χ_GI follows it.
- GI 1D mode registry: `integrator.py:52`
  `GI_MODES_1D = ['q_total','q_ip','q_oop','exit_angle']`; dispatched in
  `frame.py:641-694`; stored in `bai_1d_args['gi_mode_1d']`.

### 3.3 Headless wrapper — `xrd_tools/integrate/gid.py`
Add a dedicated wrapper beside `integrate_gi_polar_1d` (don't overload it — keep
the axis meaning unambiguous):
```
def integrate_gi_azimuthal_1d(image, fi, npt=500, npt_q=1000, method="no",
                              mask=None, radial_range=None,   # q_total band
                              azimuth_range=None,             # optional χ clip
                              incident_angle=None, tilt_angle=None,
                              sample_orientation=None, **kwargs) -> IntegrationResult1D
```
- Calls `fi.integrate1d_polar(radial_integration=True, polar_degrees=True,
  radial_unit=<A^-1|nm^-1>, npt_oop=npt, npt_ip=npt_q, ip_range=radial_range,
  oop_range=azimuth_range, sample_orientation=, incident_angle=, tilt_angle=,
  method=method, mask=mask)`.
- Read the χ_GI axis via `result.integrated` (fall back to `result.radial`) — the
  same `.integrated`-not-`.radial` handling integrate_gi_polar_1d uses (gid.py:685)
  to avoid the pyFAI warning.
- Return `IntegrationResult1D(radial=χ_GI, intensity=<count==0 → NaN>,
  unit="chigi_deg")`. NaN-fill is load-bearing: the missing wedge leaves empty
  χ bins that must be NaN, not 0.
- No "fast path" (unlike q_total): the azimuthal output always needs the polar
  path.

### 3.4 GUI touch points
- `GI_MODES_1D` (integrator.py:52): append `'chi_gi'`.
- GI `axis1D` dropdown label (near 1273-1282): add e.g. `"χ_GI (°)"`, mirroring
  `exit_angle`.
- `_NPTS_1D_DEFAULTS` (1374-1382): add `'chi_gi'` → `('500','1000')`; tooltip the
  two Pts boxes as (χ bins, q samples) — *not* the usual radial/oop meaning.
- Range field: the existing q-range (radial_range) field is the q_total band to
  integrate over — reuse directly.
- `frame.py` GI dispatch (~641-694): add
  `elif gi_mode_1d == 'chi_gi': result = integrate_gi_azimuthal_1d(...); self.gi_1d['chigi'] = result`.

---

## 4. Shared display layer (the angle axis)

Both modes output I vs a degrees angle axis (`chi_deg` / `chigi_deg`). Build this
once for Mode A; Mode B reuses it.
- x-axis label: `"χ (°)"` (Mode A) / `"χ_GI (°)"` (Mode B); range default/Auto to
  ±180° (or to the *covered* extent — GI is a partial arc).
- **NaN gaps render as gaps**, never a 0 line (partial coverage / module gaps).
- Share-Axis: an angle profile shares with nothing q-like — disable / no-op for
  these modes.
- **Reuse the path `exit_angle` already uses** for a non-q GI 1D x-axis — confirm
  how it is labeled/ranged today and follow it.
- NaN-empty helper: `gid._nan_empty_1d`/`_nan_empty_2d` are the existing
  count==0→NaN helpers. For the `single.py` standard wrapper, either import the
  helper or promote it to a tiny shared module (don't fork a copy).

### 4.1 Risks — anything that assumes a q-like 1D x-axis
Audit and guard (these break on a ±180° axis): monotonic-q / q≥0 assumptions,
Q-range filtering applied to the x-axis, Set-Bkg/baseline keyed on q, Share-Axis
pairing. `exit_angle` is the canary — wherever it works, the χ modes should too.

---

## 5. Caveats

**Shared:** the profile is a *pooled* integration (correct), but it depends on
the chosen q band, and empty azimuths (gaps, edges, GI wedge) are NaN.

**GI-specific (why Mode B is post-v1, not a free button):**
- **Which angle?** GI has ≥3 candidate "azimuths": χ_GI = arctan(q_oop/q_ip)
  (recommended, implemented), the raw detector χ (present but *not* the
  sample-frame angle under the GI transform), and the exit angles α_f/2θ_f
  (already a separate mode). Expose χ_GI explicitly; don't conflate with χ.
- **Partial, q-dependent coverage:** the missing wedge near q_oop is unmeasured →
  I(χ_GI) is an arc with NaN gaps whose span depends on the q band and incident
  angle. Auto-range to the covered extent; show gaps.
- **Conventions:** zero/sign of χ_GI depend on `sample_orientation`, `tilt_angle`,
  `polar_degrees`/`rotate`; refraction near the critical angle distorts the
  angle↔q mapping. Label the convention.
- **The "real" tool is a pole figure** — I(χ_GI) over a q band is an approximate
  1D texture cut, mixing different (q_ip,q_oop) at different angles.
- **Performance:** fiber uses `method='no'` and the polar path is
  2D-rebin-then-pool — slower than a 1D pass, fine for a per-request curve. It
  still *pools*, so it is the correct quantity; the 2D-cake oversampling/speckle
  concern does not apply to a 1D profile.

---

## 6. Tests
- **Headless (A):** `integrate_radial` wrapper returns a `chi_deg` ±180° axis,
  `.count` honored, count==0 → NaN; recovers a known cos(nχ) modulation; differs
  from a cake-row nanmean projection (confirms it's the pooled quantity);
  `radial_range` restricts the q band.
- **Headless (B):** `integrate_gi_azimuthal_1d` returns a `chigi_deg` ±180° axis,
  NaN in the unmeasured wedge, bit-identical to a direct
  `integrate1d_polar(radial_integration=True, polar_degrees=True, ...)`;
  `radial_range` restricts the q_total band.
- **GUI (both):** selecting the χ mode plots I vs angle on a ±180° axis and
  renders gaps as empty (not a 0 line); Share-Axis is inert.

---

## 7. Sequencing
1. **v1.0 — Mode A** (standard χ via `integrate_radial`): the `single.py` wrapper,
   the Units dropdown entry, the `frame.py` branch, and the shared angle-axis
   display (§4).
2. **post-v1.0 — Mode B** (GI χ_GI via `integrate1d_polar`): the
   `integrate_gi_azimuthal_1d` wrapper, the `GI_MODES_1D` entry + `frame.py`
   branch; reuses the §4 display built in step 1.
3. Add a one-line pointer to this doc from
   `docs/design/deferred_ledger.md` (the canonical deferred
   register) for Mode B.
