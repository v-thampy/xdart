"""Build the example notebooks from compact Python cell-source lists.

Each notebook is defined as a list of (cell_type, source_string) tuples;
this script converts them to .ipynb JSON.  Re-run after editing the
cell content to regenerate the notebooks.  Not packaged — for repo
maintainer use only.

    python examples/notebooks/_build_notebooks.py

All cell content has been cross-checked against the actual
``ssrl_xrd_tools`` public API (signatures + return-types) at build time
on 2026-05-17.  When the underlying API changes, edit the cell blocks
below + re-run this script.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


def md(text: str) -> tuple[str, str]:
    return ("markdown", text)


def code(text: str) -> tuple[str, str]:
    return ("code", text)


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    """Convert (type, source) tuples to nbformat 4 JSON."""
    out_cells = []
    for ctype, src in cells:
        # Splitlines keeping the trailing newline on each line, except the last.
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            pass  # final line without newline is fine
        cell: dict = {
            "cell_type": ctype,
            "metadata": {},
            "source": lines,
        }
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out_cells.append(cell)
    return {
        "cells": out_cells,
        "metadata": KERNEL_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, cells: list[tuple[str, str]]) -> None:
    nb = build_notebook(cells)
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.name}  ({len(cells)} cells)")


# =============================================================================
# 01 — Batch 1D / 2D integration
# =============================================================================

NB_BATCH_INTEGRATION = [
    md("""# Batch 1D / 2D Azimuthal Integration

Walk through the canonical batch-integration workflow in
`ssrl_xrd_tools`:

1. Load a pyFAI **PONI** calibration and a detector **mask**.
2. Iterate a directory of detector images and call
   `integrate_1d` / `integrate_2d` per frame.
3. Store the results to NeXus via `write_nexus` so they can be
   reloaded for analysis.
4. Quick sanity-check plot.

**Adapt the paths in the "Configuration" cell to your data.**
"""),
    md("## 1. Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from ssrl_xrd_tools.io import read_image, load_mask, write_nexus
from ssrl_xrd_tools.integrate import (
    load_poni,
    integrate_1d,
    integrate_2d,
)"""),
    md("## 2. Configuration\n\nAll paths and integration parameters live in this cell."),
    code("""# ───────── EDIT THESE ─────────
data_dir   = Path('~/data/my_scan').expanduser()
poni_file  = data_dir / 'calibration.poni'
mask_file  = data_dir / 'mask.edf'          # set to None if you don't have one
image_glob = '*.tif'                        # pattern relative to data_dir
out_file   = data_dir / 'integrated.nxs'

# Integration parameters
npt_1d    = 1000                            # radial bins for 1D
npt_2d    = (1000, 360)                     # (radial, azimuthal) for 2D
unit      = 'q_A^-1'                        # or '2th_deg', 'd_A'
method    = 'BBox'                          # pyFAI integration method
# ──────────────────────────────"""),
    md("## 3. Load calibration + mask"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file and mask_file.exists() else None
print(f'PONI: dist={poni.dist:.4f} m, λ={poni.wavelength * 1e10:.4f} Å')
print(f'Mask: {mask.shape if mask is not None else \"none\"}')"""),
    md("## 4. Discover images + batch integrate"),
    code("""image_files = natsorted(data_dir.glob(image_glob))
print(f'Found {len(image_files)} images matching {image_glob!r}')

results_1d, results_2d = [], []
for i, path in enumerate(image_files):
    img = read_image(path, mask=mask)
    r1 = integrate_1d(img, poni, npt=npt_1d, unit=unit, method=method, mask=mask)
    r2 = integrate_2d(img, poni, npt=npt_2d, unit=unit, method=method, mask=mask)
    results_1d.append(r1)
    results_2d.append(r2)
    if (i + 1) % 10 == 0 or i == len(image_files) - 1:
        print(f'  integrated {i + 1}/{len(image_files)}')"""),
    md("""## 5. Save to NeXus

`write_nexus` writes a single self-describing .nxs file containing all
1D + 2D results with their axes, units, and metadata. Pass the result
collections as ``results_1d=`` / ``results_2d=`` dicts keyed by frame
index."""),
    code("""results_1d_dict = {i: r for i, r in enumerate(results_1d)}
results_2d_dict = {i: r for i, r in enumerate(results_2d)}

write_nexus(
    str(out_file),
    results_1d=results_1d_dict,
    results_2d=results_2d_dict,
    overwrite=True,
)
print(f'Saved to {out_file}')"""),
    md("## 6. Sanity-check: plot first frame"),
    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

r1, r2 = results_1d[0], results_2d[0]
ax1.plot(r1.radial, r1.intensity)
ax1.set_xlabel(f'q ({r1.unit})')
ax1.set_ylabel('Intensity')
ax1.set_title('1D')

im = ax2.pcolormesh(r2.radial, r2.azimuthal, r2.intensity.T,
                    shading='auto', cmap='viridis')
ax2.set_xlabel(f'q ({r2.unit})')
ax2.set_ylabel('χ (deg)')
ax2.set_title('2D')
plt.colorbar(im, ax=ax2, label='Intensity')

plt.tight_layout()
plt.show()"""),
    md("""---

### Next steps

- For end-to-end automation (directory watching + parallel integration),
  see `ssrl_xrd_tools.integrate.process_series`.
- For multi-detector-angle scans where the detector sweeps and you want
  one stitched pattern: see notebook **02 — MultiGeometry stitching**.
- To do phase / peak fitting on the integrated results: see notebook
  **03 — Phase + peak fitting**.
"""),
]


# =============================================================================
# 02 — 1D + 2D MultiGeometry stitching
# =============================================================================

NB_STITCHING = [
    md("""# MultiGeometry Stitching (1D + 2D)

When the detector is scanned across multiple angular positions to
extend the q-range coverage, each frame has its own diffraction
geometry.  pyFAI's `MultiGeometry` machinery rebins all the frames
into one common pattern; `ssrl_xrd_tools.integrate.multi` wraps this
in two convenience functions:

* `stitch_1d(images, integrators, ...)` — combined 1D pattern.
* `stitch_2d(images, integrators, ...)` — combined 2D (q, χ) cake.

Both share a single helper `create_multigeometry_integrators` that
takes a base PONI and a per-frame list of detector rotation offsets.

This notebook walks through the full pattern using a stack of images
and a list of `rot1` (in-plane) angles.
"""),
    md("## 1. Imports + configuration"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from ssrl_xrd_tools.io import read_image, load_mask
from ssrl_xrd_tools.integrate import (
    load_poni,
    create_multigeometry_integrators,
    stitch_1d,
    stitch_2d,
)

# ───────── EDIT THESE ─────────
data_dir   = Path('~/data/my_stitched_scan').expanduser()
poni_file  = data_dir / 'calibration.poni'
mask_file  = data_dir / 'mask.edf'
image_glob = '*_scan*.tif'

# Per-image detector rotation offsets (degrees).  Length must match
# the number of images discovered below.  For a 2-circle scan this
# is typically the `del` / `tth` motor stream.
rot1_angles = np.linspace(0.0, 30.0, 11)      # 11 frames, 0° → 30° in 3° steps
rot2_angles = None                            # set to an array if you also vary rot2

# Optional per-image monitor normalisation (i0 / counts / etc.)
normalization = None                          # or e.g. np.array([...])

# Stitched-output parameters
npt_1d       = 2000
npt_rad_2d   = 1000
npt_azim_2d  = 360
unit         = 'q_A^-1'
radial_range = None                           # (qmin, qmax) or None for auto
# ──────────────────────────────"""),
    md("## 2. Load PONI + images + mask"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file.exists() else None

image_files = natsorted(data_dir.glob(image_glob))
print(f'Found {len(image_files)} frames')
assert len(image_files) == len(rot1_angles), (
    f'image count {len(image_files)} != rot1_angles length {len(rot1_angles)}'
)

images = [read_image(p, mask=mask) for p in image_files]
print(f'Loaded image stack: {len(images)} frames, '
      f'first shape {images[0].shape}')"""),
    md("""## 3. Build per-image integrators

`create_multigeometry_integrators` clones the base PONI N times and
applies the per-frame rotation offsets to each."""),
    code("""integrators = create_multigeometry_integrators(
    poni, rot1_angles, rot2_angles,
)
print(f'Built {len(integrators)} per-image integrators')
print(f'  rot1: {[f\"{np.rad2deg(ai.rot1):.2f}°\" for ai in integrators[:3]]} ...')"""),
    md("## 4. Stitch into 1D + 2D"),
    code("""result_1d = stitch_1d(
    images, integrators,
    npt=npt_1d,
    unit=unit,
    method='BBox',
    radial_range=radial_range,
    mask=mask,
    normalization=normalization,
    correctSolidAngle=False,
)

result_2d = stitch_2d(
    images, integrators,
    npt_rad=npt_rad_2d,
    npt_azim=npt_azim_2d,
    unit=unit,
    method='BBox',
    radial_range=radial_range,
    mask=mask,
    normalization=normalization,                # symmetric with stitch_1d
    correctSolidAngle=False,
)

print(f'1D: {result_1d.intensity.shape}, unit={result_1d.unit}')
print(f'2D: {result_2d.intensity.shape}, unit={result_2d.unit}')"""),
    md("## 5. Plot the stitched results"),
    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(result_1d.radial, result_1d.intensity)
ax1.set_xlabel(f'q ({result_1d.unit})')
ax1.set_ylabel('Intensity')
ax1.set_title('Stitched 1D')

im = ax2.pcolormesh(
    result_2d.radial, result_2d.azimuthal, result_2d.intensity.T,
    shading='auto', cmap='viridis',
)
ax2.set_xlabel(f'q ({result_2d.unit})')
ax2.set_ylabel('χ (deg)')
ax2.set_title('Stitched 2D')
plt.colorbar(im, ax=ax2, label='Intensity')

plt.tight_layout()
plt.show()"""),
    md("""---

### Notes

- **Per-image normalisation** (`normalization=`) is applied identically
  in both `stitch_1d` and `stitch_2d` — the same array divides each
  image before pyFAI's MultiGeometry bins them together.  Useful when
  you have per-frame monitor counts and want intensities in
  monitor-normalised units.
- For very large stacks where `images` doesn't fit in memory, the
  `StreamingGridder` machinery in `ssrl_xrd_tools.rsm.gridding` shows
  the per-chunk pattern (RSM use case, but the chunking pattern
  generalises).
"""),
]


# =============================================================================
# 03 — Phase + structure-agnostic peak fitting (single pattern)
# =============================================================================

NB_PHASE_PEAK = [
    md("""# Phase Fitting + Structure-Agnostic Peak Fitting (Single Pattern)

Two complementary fitting modes in `ssrl_xrd_tools.analysis.fitting`:

| Mode                    | Function                | When you'd use it                             |
| ----------------------- | ----------------------- | --------------------------------------------- |
| **Structure-informed**  | `PhaseFitter`           | You know what phases are present (CIF available); want phase fractions / lattice parameters |
| **Structure-agnostic**  | `fit_peaks`             | Identify / fit individual peaks without prior structural knowledge                          |

Both share the same lmfit model zoo and background utilities.  This
notebook fits the **same pattern** with both and compares results.

**Adapt the paths + `q_range` in the configuration cell.**
"""),
    md("## 1. Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssrl_xrd_tools.analysis.phase import PhaseModel
from ssrl_xrd_tools.analysis.fitting import (
    PhaseFitter,
    fit_peaks,
    extract_peaks,
    snip_1d,
)"""),
    md("## 2. Configuration"),
    code("""# ───────── EDIT THESE ─────────
data_file = Path('~/data/integrated_pattern.npz').expanduser()
cif_dir   = Path('~/data/cifs').expanduser()
cif_files = {
    'alpha':  cif_dir / 'phase_alpha.cif',
    'beta':   cif_dir / 'phase_beta.cif',
}
wavelength_A = 1.5406                  # Cu Kα — set to YOUR wavelength

q_range = (1.0, 5.0)                   # fit window in Å⁻¹
background = 'snip'                    # 'snip' | 'chebyshev_5' | 'constant' | ...
# ──────────────────────────────"""),
    md("""## 3. Load a single integrated pattern

We assume the pattern is stored as an `npz` with `q` and `intensity`
arrays.  Adapt this cell if you're reading from NeXus, XYE, etc."""),
    code("""data = np.load(data_file)
q, intensity = np.asarray(data['q']), np.asarray(data['intensity'])

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(q, intensity, lw=0.8)
ax.set_xlabel('q (Å⁻¹)'); ax.set_ylabel('Intensity')
ax.set_xlim(*q_range)
plt.tight_layout(); plt.show()"""),
    md("""## 4. Structure-informed: `PhaseFitter`

Build a `PhaseModel` per phase (peak positions + template intensities
derived from the CIF via pymatgen), feed them into a `PhaseFitter`,
fit, plot.

The q-range filter is applied when adding the phase to the fitter
(``add_phase(phase, q_range=...)``), not in ``calculate_peaks`` — the
phase carries every reflection until you trim it at fit time."""),
    code("""phases = [
    PhaseModel.from_cif(path, name=name)
    for name, path in cif_files.items()
]
for ph in phases:
    ph.calculate_peaks(wavelength=wavelength_A)
    print(f'{ph.name}: {len(ph.peaks)} reflections in CIF '
          f'(full pattern, before q-window trim)')"""),
    code("""fitter = PhaseFitter(q, intensity, prefit_background=background)
for ph in phases:
    fitter.add_phase(ph, q_range=q_range)
phase_result = fitter.fit(q_range=q_range)
print(f'PhaseFitter redχ² = {phase_result.redchi:.3f}')
print(f'Phase fractions: {phase_result.phase_fractions()}')"""),
    code("""# best_fit lives on the underlying lmfit ModelResult.  We also
# break out the per-phase components via lmfit's eval_components
# (a {component_name: ndarray} mapping).
mask = (q >= q_range[0]) & (q <= q_range[1])
best_fit = phase_result.lmfit_result.best_fit
components = phase_result.lmfit_result.eval_components()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(q[mask], intensity[mask], 'k.', ms=2, label='data')
ax.plot(q[mask], best_fit,        'C3-', lw=1.2, label='PhaseFitter')
for name, comp in components.items():
    if name.startswith('p') and '_' in name:        # p0_, p1_, ...
        ax.plot(q[mask], comp, '--', lw=0.7, label=name)
ax.set_xlabel('q (Å⁻¹)'); ax.set_ylabel('Intensity')
ax.legend(loc='best')
plt.tight_layout(); plt.show()"""),
    md("""## 5. Structure-agnostic: `fit_peaks`

`fit_peaks` accepts an explicit list of peak positions (or
auto-estimates them when ``positions=None``) and fits each as an
individual pseudo-Voigt on top of the chosen background.  Useful when
phases aren't known, or as a cross-check."""),
    code("""# A simple peak picker: SNIP-subtract then take the largest few maxima.
# In production you'd usually pass `positions=` from a manual list or
# from scipy.signal.find_peaks.
from scipy.signal import find_peaks

mask = (q >= q_range[0]) & (q <= q_range[1])
y_bs = intensity[mask] - snip_1d(intensity[mask], w=50)
idx, _ = find_peaks(y_bs, prominence=0.05 * y_bs.max(), distance=5)
positions = q[mask][idx]
print(f'Detected {len(positions)} peaks at q ≈ '
      f'{[f\"{p:.3f}\" for p in positions]}')"""),
    code("""peak_result = fit_peaks(
    q[mask], intensity[mask],
    positions=positions,
    model='pseudovoigt',
    background=background,
)
print(f'fit_peaks redχ² = {peak_result.fit_result.redchi:.3f}')
print(f'Fit {peak_result.n_peaks} peaks ({peak_result.model_name} on '
      f'{peak_result.background_name} background)')
print('First 3 peaks (centre / sigma / amplitude):')
for c, s, a in list(zip(peak_result.peak_centers,
                        peak_result.peak_sigmas,
                        peak_result.peak_amplitudes))[:3]:
    print(f'  q={c:.4f}  σ={s:.4f}  amp={a:.2f}')"""),
    md("## 6. Compare side-by-side"),
    code("""fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
ax, axr = axes
ax.plot(q[mask], intensity[mask], 'k.', ms=2, label='data')
ax.plot(q[mask], best_fit,             'C3-', lw=1.0,
        label=f'PhaseFitter (redχ²={phase_result.redchi:.2f})')
ax.plot(q[mask], peak_result.best_fit, 'C0-', lw=1.0,
        label=f'fit_peaks (redχ²={peak_result.fit_result.redchi:.2f})')
ax.set_ylabel('Intensity'); ax.legend()

axr.plot(q[mask], intensity[mask] - best_fit,             'C3-', lw=0.6,
         label='PhaseFitter residual')
axr.plot(q[mask], intensity[mask] - peak_result.best_fit, 'C0-', lw=0.6,
         label='fit_peaks residual')
axr.axhline(0, color='k', lw=0.5)
axr.set_xlabel('q (Å⁻¹)'); axr.set_ylabel('residual'); axr.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""---

### When to prefer which

- **`PhaseFitter`** — when phase identification is the goal (relative
  fractions, lattice parameter trends across a sample series).  See
  notebook **04 — Batch phase fitting** for the sequence pattern.
- **`fit_peaks`** — when you care about individual peak positions /
  widths / areas (peak shifts vs. temperature, line broadening
  analysis).
"""),
]


# =============================================================================
# 04 — Batch phase fitting
# =============================================================================

NB_BATCH_PHASE = [
    md("""# Batch Phase Fitting

Run `PhaseFitter` over a *sequence* of integrated patterns using the
`FitConfig` + `fit_sequence` + `FitResultStore` pipeline in
`ssrl_xrd_tools.analysis.fitting.batch`.

This is the headless equivalent of xdart's `BatchPhaseFitViewer`.
The result is a `pandas.DataFrame` with one row per pattern, ready
for plotting trends (phase fractions vs. position / temperature /
composition).
"""),
    md("## 1. Imports + paths"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ssrl_xrd_tools.analysis.phase import PhaseModel
from ssrl_xrd_tools.analysis.fitting import FitConfig, fit_sequence

# ───────── EDIT THESE ─────────
data_dir    = Path('~/data/HZO_series').expanduser()
patterns_pkl = data_dir / 'patterns.npz'   # or however you store them
cif_dir     = data_dir / 'cifs'
out_csv     = data_dir / 'phase_fractions.csv'

wavelength_A = 1.5406
q_range      = (1.5, 5.5)
# ──────────────────────────────"""),
    md("""## 2. Load patterns

`fit_sequence` accepts a list of ``(q, intensity)`` tuples (or
``(q, intensity, sigma)`` if you have per-point errors).  Here we
assume an `.npz` with arrays ``q``, ``intensities`` (shape
``(N_patterns, len(q))``) and optionally ``labels``.  Adapt this
cell if you're loading from NeXus, separate files, etc."""),
    code("""npz = np.load(patterns_pkl, allow_pickle=True)
q              = np.asarray(npz['q'])
intensities    = np.asarray(npz['intensities'])      # (N, len(q))
labels         = list(npz['labels']) if 'labels' in npz else \\
                 [f'frame_{i:04d}' for i in range(intensities.shape[0])]
patterns       = [(q, intensities[i]) for i in range(intensities.shape[0])]
print(f'Loaded {len(patterns)} patterns; q range '
      f'{q.min():.3f} → {q.max():.3f} Å⁻¹')"""),
    md("## 3. Define phases"),
    code("""phases = [
    PhaseModel.from_cif(cif_dir / 'phase_alpha.cif', name='alpha'),
    PhaseModel.from_cif(cif_dir / 'phase_beta.cif',  name='beta'),
]
for ph in phases:
    ph.calculate_peaks(wavelength=wavelength_A)
    print(f'  {ph.name}: {len(ph.peaks)} reflections (will be q-trimmed at fit time)')"""),
    md("""## 4. Build a FitConfig

`FitConfig` captures **two** keyword dicts:

- ``init_kw`` — passed to `PhaseFitter.__init__` (background settings,
  amorphous-peak settings, in-fit background, …).
- ``fit_kw`` — passed to `PhaseFitter.fit()` (q_range, profile,
  Caglioti toggles, texture, lattice/width bounds, …).

Plus ``phase_names`` (which phases from the list to include) and
``min_intensity`` (template-intensity floor for `add_phase`).  The
whole thing round-trips through JSON."""),
    code("""config = FitConfig(
    init_kw={
        'prefit_background': 'snip',
    },
    fit_kw={
        'q_range': q_range,
        'method': 'leastsq',
        'lattice_pct': 0.05,           # ±5% lattice-parameter bounds
        'phase_profile': 'pseudovoigt',
    },
    phase_names=[ph.name for ph in phases],
    min_intensity=5.0,
    name='HZO series — first pass',
)
config.save(data_dir / 'fit_config.json')
print(f'Saved config → {data_dir / \"fit_config.json\"}')
# Reload elsewhere:  config = FitConfig.load(data_dir / 'fit_config.json')"""),
    md("## 5. Run the sequence"),
    code("""def _progress(i: int, n: int, result) -> None:
    if (i + 1) % 5 == 0 or i == n - 1:
        ok = 'OK' if result.success else 'FAIL'
        print(f'  [{i+1:3d}/{n}] redχ²={result.redchi:.3f}  {ok}')

store = fit_sequence(
    patterns,
    phases,
    config,
    labels=labels,
    progress_callback=_progress,
)
print(f'\\nCompleted {len(store)} fits')"""),
    md("## 6. Results as DataFrame"),
    code("""df = store.to_dataframe()
print(df.head().to_string(index=False))
df.to_csv(out_csv, index=False)
print(f'\\nSaved → {out_csv}')"""),
    md("## 7. Plot phase fractions"),
    code("""frac_cols = [c for c in df.columns if c.startswith('frac_')]

fig, ax = plt.subplots(figsize=(9, 4))
for col in frac_cols:
    ax.plot(df['index'], df[col], 'o-', label=col.replace('frac_', ''))
ax.set_xlabel('pattern index')
ax.set_ylabel('phase fraction')
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout(); plt.show()"""),
    md("""## 8. (Optional) lattice-parameter trends

If `vary_cell_params=True` was set, the DataFrame also has columns
like `alpha_a`, `alpha_b`, `alpha_c`, etc."""),
    code("""cell_cols = [c for c in df.columns
             if any(c.endswith('_' + x) for x in ('a', 'b', 'c'))]
if cell_cols:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, axis in zip(axes, ['a', 'b', 'c']):
        for ph_name in [p.name for p in phases]:
            col = f'{ph_name}_{axis}'
            if col in df.columns:
                ax.plot(df['index'], df[col], 'o-', label=ph_name)
        ax.set_ylabel(f'{axis} (Å)')
        ax.set_xlabel('pattern index')
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print('No lattice-parameter columns — set vary_cell_params=True in FitConfig.')"""),
    md("""---

### Re-running with new parameters

The whole pipeline is config-driven, so re-running with different
background / q_range / lattice flags is one config-edit + one
`fit_sequence` call.  The `FitResultStore` overwrites its rows on
re-fit; serialise it (`store.save(...)`) if you want to keep multiple
runs side-by-side.
"""),
]


# =============================================================================
# 05 — sin²ψ strain analysis
# =============================================================================

NB_SIN2PSI = [
    md("""# Grazing-Incidence sin²ψ Strain Analysis

Compute biaxial strain (and optionally stress) from a single
grazing-incidence detector image using the `sin²ψ` method:

1. **GI polar integration** — `integrate_gi_polar` produces an
   `(q, χ)` map corrected for the grazing-incidence geometry.
2. **χ-sector extraction** — `extract_chi_sectors` slices the map
   into ψ-binned 1D profiles.
3. **Per-sector peak fit** — `fit_peak_vs_psi` fits the same peak in
   every sector, returning q_peak(ψ).
4. **Regression** — `sin2psi_regression` fits
   `d(ψ) = d₀ + slope · sin²ψ`; slope/d₀ → strain; optional `(E, ν)`
   → stress.

Or, one-call: `sin2psi_analysis(...)` does all four steps.
"""),
    md("## 1. Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssrl_xrd_tools.io.image import read_image, load_mask
from ssrl_xrd_tools.io.metadata import read_txt_metadata
from ssrl_xrd_tools.integrate import load_poni, integrate_gi_polar
from ssrl_xrd_tools.integrate.calibration import poni_to_fiber_integrator
from ssrl_xrd_tools.analysis.strain import (
    extract_chi_sectors,
    fit_peak_vs_psi,
    sin2psi_regression,
    sin2psi_analysis,
)"""),
    md("## 2. Configuration"),
    code("""# ───────── EDIT THESE ─────────
data_path = Path('~/data/sin2psi').expanduser()
poni_file = data_path / 'calibration.poni'
data_file = data_path / 'sample_asDep.tif'
meta_file = data_path / 'sample_asDep.txt'   # incidence angle, etc.
mask_file = data_path / 'mask.edf'

# Peak of interest: q-window around the (hkl) being analysed.
q_range = (3.0, 3.48)

# Sector configuration
chi_width  = 5.0                          # degrees per ψ bin
n_sectors  = 15                           # number of ψ bins

# Optional elastic constants for stress output (same units as the
# returned stress — use GPa here and you'll get GPa back).
E   = 70.0                                # Young's modulus
nu  = 0.34                                # Poisson's ratio
# ──────────────────────────────"""),
    md("## 3. Load image + build FiberIntegrator"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file.exists() else None
data = read_image(data_file, mask=mask)
meta = read_txt_metadata(meta_file) if meta_file.exists() else {}

# The fiber integrator handles GI geometry corrections
incidence_angle = float(meta.get('th', 0.5))     # incidence angle in degrees
fi = poni_to_fiber_integrator(poni, incidence_angle=incidence_angle)
print(f'Loaded image {data.shape}, incidence = {incidence_angle}°')"""),
    md("## 4. GI polar map (q_total vs. χ)"),
    code("""polar = integrate_gi_polar(
    data, fi,
    npt_rad=800,
    npt_azim=360,
    mask=mask,
    radial_range=q_range,
)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.pcolormesh(
    polar.radial, polar.azimuthal, polar.intensity.T,
    shading='auto', cmap='viridis',
)
ax.set_xlabel(f'q ({polar.unit})'); ax.set_ylabel('χ (deg)')
ax.set_title('GI polar map')
plt.colorbar(im, ax=ax, label='Intensity')
plt.tight_layout(); plt.show()"""),
    md("""## 5. Extract χ sectors + fit the peak in each

`fit_peak_vs_psi` returns a list of `PeakFitResult` objects; each
carries `psi`, `sin2psi`, `q_center`, `d_spacing`, and their 1-σ
uncertainties."""),
    code("""sectors = extract_chi_sectors(
    polar,
    chi_width=chi_width,
    n_sectors=n_sectors,
)
print(f'Extracted {len(sectors)} ψ sectors')

peak_fits = fit_peak_vs_psi(
    sectors,
    q_range=q_range,
    model='pseudovoigt',
    background='linear',
)
psis = np.array([pf.psi      for pf in peak_fits])
qcs  = np.array([pf.q_center for pf in peak_fits])
qerr = np.array([pf.q_center_err for pf in peak_fits])
print(f'  ψ range : {psis.min():.1f}° → {psis.max():.1f}°')
print(f'  q_peak  : {qcs.min():.4f} → {qcs.max():.4f} Å⁻¹')"""),
    md("""## 6. sin²ψ regression → d₀, slope, optional stress

`sin2psi_regression` takes the list of `PeakFitResult` (NOT raw
arrays).  Supply `E` and `nu` to get a stress value back in the same
units as `E`."""),
    code("""result = sin2psi_regression(peak_fits, E=E, nu=nu)
print(f'  d_0     = {result.d0:.6f} ± {result.d0_err:.6f} Å')
print(f'  slope   = {result.slope:.4e} ± {result.slope_err:.4e} Å / sin²ψ')
print(f'  R²      = {result.r_squared:.4f}')
if result.stress is not None:
    print(f'  stress  = {result.stress:.2f} ± {result.stress_err:.2f} '
          f'(units of E, here GPa)')
# Strain along the in-plane direction is conventionally (slope / d0)
print(f'  ε(sin²ψ→1) ≈ slope / d_0 = {result.slope / result.d0 * 100:.3f} %')"""),
    md("## 7. Diagnostic plot"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# d vs sin²ψ — the regression
ax = axes[0]
ax.errorbar(result.sin2psi, result.d_values,
            yerr=result.d_errors, fmt='o', label='per-sector')
fitline = np.linspace(result.sin2psi.min(), result.sin2psi.max(), 50)
ax.plot(fitline, result.d0 + result.slope * fitline, 'r-',
        label=f'fit:  d = {result.d0:.4f}{result.slope:+.2e}·sin²ψ')
ax.set_xlabel('sin²ψ'); ax.set_ylabel('d (Å)')
ax.set_title(f'sin²ψ regression  (R²={result.r_squared:.3f})')
ax.legend(fontsize=9)

# Per-sector q vs ψ
ax = axes[1]
ax.errorbar(psis, qcs, yerr=qerr, fmt='o-')
ax.set_xlabel('ψ (deg)'); ax.set_ylabel('q_center (Å⁻¹)')
ax.set_title('peak position vs ψ')

plt.tight_layout(); plt.show()"""),
    md("""## 8. One-call shortcut

`sin2psi_analysis(result2d, q_range, ...)` chains the previous four
steps.  Note it takes the **polar map** (`IntegrationResult2D`) — not
the raw image — so the GI integration step still lives outside."""),
    code("""one_shot = sin2psi_analysis(
    polar,
    q_range=q_range,
    chi_width=chi_width,
    n_sectors=n_sectors,
    model='pseudovoigt',
    background='linear',
    E=E, nu=nu,
)
print(f'one-call d_0    = {one_shot.d0:.6f} Å')
print(f'one-call slope  = {one_shot.slope:.4e}')
print(f'one-call stress = {one_shot.stress}'
      f' (units of E={E})')"""),
    md("""---

### Notes

- All four steps are individually scriptable, so a non-standard
  pipeline (per-sector peak constraints, multiple peaks per sector,
  film/substrate separation) is straightforward to assemble from the
  primitives.
- For multi-peak strain analysis, run `sin2psi_analysis` per peak and
  aggregate the regression slopes.
"""),
]


# =============================================================================
# Build everything
# =============================================================================

if __name__ == "__main__":
    write_notebook("01_batch_integration.ipynb",      NB_BATCH_INTEGRATION)
    write_notebook("02_multigeometry_stitching.ipynb", NB_STITCHING)
    write_notebook("03_phase_and_peak_fitting.ipynb",  NB_PHASE_PEAK)
    write_notebook("04_batch_phase_fitting.ipynb",     NB_BATCH_PHASE)
    write_notebook("05_sin2psi_analysis.ipynb",        NB_SIN2PSI)
    print("\nDone.  Re-run after editing cell content above.")
