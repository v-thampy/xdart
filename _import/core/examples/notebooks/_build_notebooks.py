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
    md("## Imports"),
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
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom.
Nothing in the later cells needs changing — the validation cell catches
typos before integration starts."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_dir   = Path('~/data/my_scan').expanduser()    # ← REPLACE
poni_file  = data_dir / 'calibration.poni'           # ← REPLACE
image_glob = '*.tif'                                  # pattern inside data_dir

# ── OPTIONAL paths ─────────────────────────────────────────────────────
mask_file  = data_dir / 'mask.edf'           # set to None if no mask file
out_file   = data_dir / 'integrated.nxs'

# ── Integration tuning ─────────────────────────────────────────────────
npt_1d  = 1000                # 1D radial bins
npt_2d  = (1000, 360)          # (radial, azimuthal) for 2D
unit    = 'q_A^-1'             # 'q_A^-1' | '2th_deg' | 'd_A'
method  = 'BBox'               # pyFAI integration method"""),
    md("### Validate the config"),
    code("""assert data_dir.is_dir(), f'data_dir not found: {data_dir}'
assert poni_file.is_file(), f'poni_file not found: {poni_file}'
if mask_file is not None:
    assert mask_file.is_file(), f'mask_file not found: {mask_file}'
_images_preview = sorted(data_dir.glob(image_glob))
assert _images_preview, f'No images match {image_glob!r} in {data_dir}'
print(f'OK — {len(_images_preview)} image(s) found, PONI present.')"""),
    md("## Load calibration + mask"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file and mask_file.exists() else None
print(f'PONI: dist={poni.dist:.4f} m, λ={poni.wavelength * 1e10:.4f} Å')
print(f'Mask: {mask.shape if mask is not None else \"none\"}')"""),
    md("## Discover images + batch integrate"),
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
    md("""## Save to NeXus

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
    md("## Sanity-check: plot first frame"),
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

- **Notebook 06 — Headless Reduction Pipeline** is the recommended
  higher-level alternative to the per-frame `integrate_1d` /
  `integrate_2d` loop above.  It uses `ReductionPlan` + `Scan` +
  `Frame` + `NexusSink` so the same configuration can drive notebook,
  CLI, and the xdart GUI without code changes.  Prefer it for
  new code; this notebook stays as the "low-level building blocks"
  reference.
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
    md("## Imports"),
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
)"""),
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_dir   = Path('~/data/my_stitched_scan').expanduser()  # ← REPLACE
poni_file  = data_dir / 'calibration.poni'                  # ← REPLACE
image_glob = '*_scan*.tif'                                   # pattern in data_dir

# Per-image detector rotation offsets (degrees).  Length must match
# the number of images discovered.  For a 2-circle scan this is
# typically the `del` / `tth` motor stream.
rot1_angles = np.linspace(0.0, 30.0, 11)     # ← REPLACE — 11 frames, 0° → 30°
rot2_angles = None                            # set to an array if rot2 also varies

# ── OPTIONAL ───────────────────────────────────────────────────────────
mask_file     = data_dir / 'mask.edf'        # set to None if no mask
normalization = None                          # or np.array([...]) of per-frame i0

# ── Stitched-output tuning ─────────────────────────────────────────────
npt_1d       = 2000
npt_rad_2d   = 1000
npt_azim_2d  = 360
unit         = 'q_A^-1'
radial_range = None                          # (qmin, qmax) or None for auto"""),
    md("### Validate the config"),
    code("""assert data_dir.is_dir(), f'data_dir not found: {data_dir}'
assert poni_file.is_file(), f'poni_file not found: {poni_file}'
if mask_file is not None:
    assert mask_file.is_file(), f'mask_file not found: {mask_file}'
_images_preview = natsorted(data_dir.glob(image_glob))
assert _images_preview, f'No images match {image_glob!r} in {data_dir}'
assert len(_images_preview) == len(rot1_angles), (
    f'image count {len(_images_preview)} != rot1_angles length {len(rot1_angles)}'
)
print(f'OK — {len(_images_preview)} frame(s), PONI present.')"""),
    md("## Load PONI + images + mask"),
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
    md("""## Build per-image integrators

`create_multigeometry_integrators` clones the base PONI N times and
applies the per-frame rotation offsets to each."""),
    code("""integrators = create_multigeometry_integrators(
    poni, rot1_angles, rot2_angles,
)
print(f'Built {len(integrators)} per-image integrators')
print(f'  rot1: {[f\"{np.rad2deg(ai.rot1):.2f}°\" for ai in integrators[:3]]} ...')"""),
    md("## Stitch into 1D + 2D"),
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
    md("## Plot the stitched results"),
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
    md("## Imports"),
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
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_file = Path('~/data/integrated_pattern.npz').expanduser()  # ← REPLACE
cif_dir   = Path('~/data/cifs').expanduser()                    # ← REPLACE
cif_files = {
    'alpha':  cif_dir / 'phase_alpha.cif',     # ← REPLACE with your phases
    'beta':   cif_dir / 'phase_beta.cif',
}
wavelength_A = 1.5406              # ← REPLACE — Cu Kα default, set to YOUR wavelength

# ── Fitting parameters ─────────────────────────────────────────────────
q_range    = (1.0, 5.0)            # fit window in Å⁻¹
background = 'snip'                # 'snip' | 'chebyshev_5' | 'constant' | ..."""),
    md("### Validate the config"),
    code("""assert data_file.is_file(), f'data_file not found: {data_file}'
assert cif_dir.is_dir(), f'cif_dir not found: {cif_dir}'
for name, path in cif_files.items():
    assert path.is_file(), f'CIF {name!r} not found: {path}'
print(f'OK — pattern + {len(cif_files)} CIF(s) present.')"""),
    md("""## Load a single integrated pattern

We assume the pattern is stored as an `npz` with `q` and `intensity`
arrays.  Adapt this cell if you're reading from NeXus, XYE, etc."""),
    code("""data = np.load(data_file)
q, intensity = np.asarray(data['q']), np.asarray(data['intensity'])

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(q, intensity, lw=0.8)
ax.set_xlabel('q (Å⁻¹)'); ax.set_ylabel('Intensity')
ax.set_xlim(*q_range)
plt.tight_layout(); plt.show()"""),
    md("""## Structure-informed: `PhaseFitter`

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
    md("""## Structure-agnostic: `fit_peaks`

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
    md("## Compare side-by-side"),
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
    md("## Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ssrl_xrd_tools.analysis.phase import PhaseModel
from ssrl_xrd_tools.analysis.fitting import FitConfig, fit_sequence"""),
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_dir     = Path('~/data/HZO_series').expanduser()  # ← REPLACE
patterns_pkl = data_dir / 'patterns.npz'                # ← REPLACE
cif_dir      = data_dir / 'cifs'                        # ← REPLACE

wavelength_A = 1.5406              # ← REPLACE — your beamline wavelength in Å

# ── OPTIONAL ───────────────────────────────────────────────────────────
out_csv  = data_dir / 'phase_fractions.csv'

# ── Fit window ─────────────────────────────────────────────────────────
q_range  = (1.5, 5.5)"""),
    md("### Validate the config"),
    code("""assert data_dir.is_dir(), f'data_dir not found: {data_dir}'
assert patterns_pkl.is_file(), f'patterns_pkl not found: {patterns_pkl}'
assert cif_dir.is_dir(), f'cif_dir not found: {cif_dir}'
print(f'OK — patterns + CIF dir present.')"""),
    md("""## Load patterns

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
    md("## Define phases"),
    code("""phases = [
    PhaseModel.from_cif(cif_dir / 'phase_alpha.cif', name='alpha'),
    PhaseModel.from_cif(cif_dir / 'phase_beta.cif',  name='beta'),
]
for ph in phases:
    ph.calculate_peaks(wavelength=wavelength_A)
    print(f'  {ph.name}: {len(ph.peaks)} reflections (will be q-trimmed at fit time)')"""),
    md("""## Build a FitConfig

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
    md("## Run the sequence"),
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
    md("## Results as DataFrame"),
    code("""df = store.to_dataframe()
print(df.head().to_string(index=False))
df.to_csv(out_csv, index=False)
print(f'\\nSaved → {out_csv}')"""),
    md("## Plot phase fractions"),
    code("""frac_cols = [c for c in df.columns if c.startswith('frac_')]

fig, ax = plt.subplots(figsize=(9, 4))
for col in frac_cols:
    ax.plot(df['index'], df[col], 'o-', label=col.replace('frac_', ''))
ax.set_xlabel('pattern index')
ax.set_ylabel('phase fraction')
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout(); plt.show()"""),
    md("""## (Optional) lattice-parameter trends

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
    md("## Imports"),
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
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_path = Path('~/data/sin2psi').expanduser()    # ← REPLACE
poni_file = data_path / 'calibration.poni'           # ← REPLACE
data_file = data_path / 'sample_asDep.tif'           # ← REPLACE

# Peak of interest: q-window around the (hkl) being analysed.
q_range = (3.0, 3.48)                                # ← REPLACE

# ── OPTIONAL paths ─────────────────────────────────────────────────────
meta_file = data_path / 'sample_asDep.txt'   # incidence angle, etc.
mask_file = data_path / 'mask.edf'

# ── Sector configuration ──────────────────────────────────────────────
chi_width  = 5.0                          # degrees per ψ bin
n_sectors  = 15                           # number of ψ bins

# ── Elastic constants (optional — for stress output) ──────────────────
# Units of E set the units of the returned stress.
E   = 70.0                                # Young's modulus, GPa
nu  = 0.34                                # Poisson's ratio"""),
    md("### Validate the config"),
    code("""assert data_path.is_dir(), f'data_path not found: {data_path}'
assert poni_file.is_file(), f'poni_file not found: {poni_file}'
assert data_file.is_file(), f'data_file not found: {data_file}'
if mask_file and mask_file.exists():
    pass
else:
    mask_file = None     # silently disable if missing
if meta_file and not meta_file.exists():
    meta_file = None
print(f'OK — image + PONI present; '
      f'mask={\"yes\" if mask_file else \"no\"}, '
      f'meta={\"yes\" if meta_file else \"no\"}.')"""),
    md("## Load image + build FiberIntegrator"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file.exists() else None
data = read_image(data_file, mask=mask)
meta = read_txt_metadata(meta_file) if meta_file.exists() else {}

# The fiber integrator handles GI geometry corrections
incidence_angle = float(meta.get('th', 0.5))     # incidence angle in degrees
fi = poni_to_fiber_integrator(poni, incidence_angle=incidence_angle)
print(f'Loaded image {data.shape}, incidence = {incidence_angle}°')"""),
    md("## GI polar map (q_total vs. χ)"),
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
    md("""## Extract χ sectors + fit the peak in each

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
    md("""## sin²ψ regression → d₀, slope, optional stress

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
    md("## Diagnostic plot"),
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
    md("""## One-call shortcut

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

# =============================================================================
# 06 — Headless reduction pipeline (ReductionPlan + run_reduction + sinks)
# =============================================================================

NB_REDUCTION_PIPELINE = [
    md("""# Headless Reduction Pipeline

The canonical headless reduction workflow in `ssrl_xrd_tools.reduction`:

1. Wrap your data in a **`Scan`** of **`Frame`** objects (one per
   detector image, with optional lazy loading).
2. Describe what to do in a **`ReductionPlan`** (1D / 2D integration
   settings, optional GI mode, mask, threshold).
3. Pick a **sink** — `MemorySink` for in-notebook follow-up,
   `NexusSink` for persistent batch processing, or your own custom
   sink (e.g. a Qt signal sink for GUI streaming).
4. Run **`run_reduction(plan, scan, sink, *, chunk_size, progress_cb,
   cancel_token)`** — same call works in a notebook, a CLI, or
   xdart's wrangler threads.

This is the API the xdart wranglers were rewritten on top of —
demonstrating it here in a notebook gives you the same canonical
path that the GUI uses, without the GUI.
"""),
    md("## Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from ssrl_xrd_tools.io import load_mask, read_image
from ssrl_xrd_tools.integrate import load_poni
from ssrl_xrd_tools.reduction import (
    CancelToken,
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    MemorySink,
    NexusSink,
    ReductionPlan,
    ReductionProgress,
    Scan,
    run_reduction,
)"""),
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest of the notebook top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — paths + parameters                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
data_dir   = Path('~/data/my_scan').expanduser()    # ← REPLACE
poni_file  = data_dir / 'calibration.poni'           # ← REPLACE
image_glob = '*.tif'                                  # pattern inside data_dir

# ── OPTIONAL paths ─────────────────────────────────────────────────────
mask_file  = data_dir / 'mask.edf'           # set to None if no mask
out_nxs    = data_dir / 'reduced.nxs'

# ── Integration tuning ─────────────────────────────────────────────────
npt_1d                   = 1000
npt_2d_rad, npt_2d_azim  = 1000, 360
unit                     = 'q_A^-1'
method                   = 'csr'

# ── Grazing incidence (optional) ──────────────────────────────────────
# Set to None for transmission geometry.  Uncomment + set for GI.
gi: GIMode | None = None
# gi = GIMode(incident_angle=0.5)

# ── Monitor normalisation (optional) ──────────────────────────────────
# Key in each Frame.metadata that holds the i0/monitor reading.
monitor_key: str | None = None       # e.g. 'i0'"""),
    md("### Validate the config"),
    code("""assert data_dir.is_dir(), f'data_dir not found: {data_dir}'
assert poni_file.is_file(), f'poni_file not found: {poni_file}'
if mask_file is not None and not mask_file.exists():
    mask_file = None                 # silently disable if missing
_images_preview = sorted(data_dir.glob(image_glob))
assert _images_preview, f'No images match {image_glob!r} in {data_dir}'
print(f'OK — {len(_images_preview)} image(s) found, PONI present.')"""),
    md("## Build the `Scan` + `Frame` list"),
    code("""poni = load_poni(poni_file)
mask = load_mask(mask_file) if mask_file.exists() else None

# Lazy loading: pass source_path instead of image=, and Frame.load_image()
# reads the file on demand inside run_reduction.  Use clear_frame_images=
# below to release each frame after it's been written to the sink.
image_files = natsorted(data_dir.glob(image_glob))
print(f'Found {len(image_files)} images')

frames = [
    Frame(
        index=i,
        source_path=path,
        # metadata can carry per-frame counters that monitor_key picks up
        metadata={'i0': 1.0},   # replace with your actual i0 readouts
    )
    for i, path in enumerate(image_files)
]

scan = Scan(
    name='my_scan',
    frames=frames,
    poni=poni,
    output_path=out_nxs,
)
print(f'Built Scan with {len(scan)} frames')"""),
    md("""## Compose a `ReductionPlan`

The plan captures *what* the reduction does — integration settings,
mask, threshold, GI mode.  Execution-policy knobs like `chunk_size` /
`clear_frame_images` are arguments to `run_reduction`, not part of the
plan, so the same plan can be re-run on different scans with
different chunking."""),
    code("""plan = ReductionPlan(
    integration_1d=Integration1DPlan(
        npt=npt_1d,
        unit=unit,
        method=method,
        monitor_key=monitor_key,
    ),
    integration_2d=Integration2DPlan(
        npt_rad=npt_2d_rad,
        npt_azim=npt_2d_azim,
        unit=unit,
        method=method,
        monitor_key=monitor_key,
    ),
    # MaskSpec lets you pass a flat-index mask without knowing the
    # detector shape upfront; the executor resolves it when the first
    # frame is loaded.  Pass a plain 2D bool array to skip the wrapper.
    mask=MaskSpec(mask) if mask is not None else None,
    threshold_min=None,                # e.g. 0 to clip negative dark
    threshold_max=None,                # e.g. 1e8 to clip hot pixels
    gi=gi,                             # None for transmission, GIMode(...) for GI
)
print(plan)"""),
    md("""## Run with the in-memory sink

`MemorySink` collects every frame's `FrameReduction` into a dict —
handy in notebooks where you want to inspect / plot the results
immediately."""),
    code("""sink = MemorySink()
progress_events: list[ReductionProgress] = []

result = run_reduction(
    plan, scan, sink,
    chunk_size=8,                    # tune to your detector + memory
    clear_frame_images=True,         # release each frame's image after sink.write
    progress_cb=progress_events.append,
)

print(f'Processed {result.n_processed}/{len(scan)} frames')
print(f'Last 3 progress events: {progress_events[-3:]}')
first = sink.frames[0]
print(f'Frame 0: 1D shape {first.result_1d.intensity.shape}, '
      f'2D shape {first.result_2d.intensity.shape}')"""),
    md("## Re-run with a `NexusSink` for persistent batch output"),
    code("""# Same plan + scan, different sink.  The .nxs file is written
# frame-by-frame; flush_every=16 means the OS flushes after every
# 16 frames so a crashed run is mostly recoverable.
nx_sink = NexusSink(out_nxs, overwrite=True, flush_every=16)

result = run_reduction(
    plan, scan, nx_sink,
    chunk_size=8,
    clear_frame_images=True,
)
print(f'Wrote {result.n_processed} frames to {result.output_path}')"""),
    md("""## Cancellation

Any consumer can flip the cancel token to stop the loop at the next
frame boundary.  Useful for GUI Stop buttons / Ctrl-C in notebooks."""),
    code("""token = CancelToken()

# Simulate a cancel after 3 frames have been processed.
def _cancel_at_3(event: ReductionProgress) -> None:
    if event.stage == 'write' and event.completed >= 3:
        token.cancel()

partial = run_reduction(
    plan, scan, MemorySink(),
    chunk_size=4,
    progress_cb=_cancel_at_3,
    cancel_token=token,
)
print(f'Cancelled? {partial.cancelled}; '
      f'processed {partial.n_processed} of {len(scan)}')"""),
    md("## Plot the first frame's results"),
    code("""r1 = sink.frames[0].result_1d
r2 = sink.frames[0].result_2d

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(r1.radial, r1.intensity)
ax1.set_xlabel(f'q ({r1.unit})')
ax1.set_ylabel('Intensity')

im = ax2.pcolormesh(r2.radial, r2.azimuthal, r2.intensity.T,
                    shading='auto', cmap='viridis')
ax2.set_xlabel(f'q ({r2.unit})')
ax2.set_ylabel('χ (deg)')
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()"""),
    md("""---

### Notes

- **GI workflow**: set `gi=GIMode(incident_angle=...)` on the plan;
  the executor builds a `FiberIntegrator` from `scan.poni` + the
  GIMode parameters and routes through `integrate_gi_1d` /
  `integrate_gi_2d`.
- **Custom sinks**: any class with `begin(scan, plan)` /
  `write(frame, reduction)` / `finish(result)` methods is a valid
  `ReductionSink` (it's a `typing.Protocol`).  xdart adds its own
  Qt-signal sink on top of this; you can do the same with a plain
  Python collector, a Tiled writer, a CSV row appender, etc.
- **Same code in xdart**: the GUI's wrangler threads build a
  `ReductionPlan` from sphere settings and call `run_reduction` —
  the same workflow you just ran.  See the `keep_xdart_thin.md`
  memory note for the design rationale.
"""),
]


# =============================================================================
# 07 — Reading processed .nxs files (convenience readers)
# =============================================================================

NB_READING_PROCESSED = [
    md("""# Reading processed `.nxs` files

Once a scan has been reduced (by the xdart GUI, by notebook 01, or by
the headless pipeline in notebook 06) the results live in a v2 NeXus
`.nxs` file.  This notebook shows the **simplest possible way** to pull
1D and 2D integrated patterns back out for plotting or analysis, using
the `get_*` convenience readers in `ssrl_xrd_tools.io`.

A processed file is a **scan**: a stack of integrated **frames**.  Each
reader takes a `frame` label (the value stored in the file, e.g. 1-based
for SPEC, possibly gapped) or `None` for all frames.

For the full `xarray.Dataset` (every motor column + provenance) use
`read_sphere` / `read_sphere_metadata` instead — but for "open a file,
get arrays I can plot" the helpers below are all you need."""),
    md("## Imports"),
    code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssrl_xrd_tools.io import (
    get_frames,
    get_metadata,
    get_1d,
    get_2d,
    get_thumbnail,
    open_scan,
)"""),
    md("""## ✏️ Configuration

**Edit the cell below**, then run the rest top-to-bottom."""),
    code("""# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EDIT THIS CELL — path + frame to inspect             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── REQUIRED ───────────────────────────────────────────────────────────
scan_file = Path('~/data/my_scan/integrated.nxs').expanduser()   # ← REPLACE

# ── OPTIONAL ───────────────────────────────────────────────────────────
# Which single frame to spotlight in the per-frame plots below.  Leave as
# None to auto-pick the first frame in the file.
frame_to_show = None"""),
    md("### Validate the config"),
    code("""assert scan_file.is_file(), f'scan_file not found: {scan_file}'
assert scan_file.suffix == '.nxs', f'expected a .nxs file, got {scan_file.suffix}'
print(f'OK — reading {scan_file.name}')"""),
    md("""## What's in the file?

`get_frames` lists the frame labels; `get_metadata` returns a flat dict
of scan-level info (no heavy intensity arrays loaded)."""),
    code("""frames = get_frames(scan_file)
meta = get_metadata(scan_file)

print(f'frames ({meta[\"n_frames\"]}): {frames.tolist()}')
print(f'sample : {meta[\"sample_name\"] or \"(unnamed)\"}')
print(f'energy : {meta[\"energy_keV\"]:.4f} keV   λ = {meta[\"wavelength_A\"]:.5f} Å')
print(f'has 1D : {meta[\"has_1d\"]}    has 2D : {meta[\"has_2d\"]}')
print(f'motors : {list(meta[\"positioners\"].keys())}')

# Pick the frame to spotlight (first one if not set above)
spotlight = frame_to_show if frame_to_show is not None else int(frames[0])
print(f'\\nSpotlighting frame {spotlight}')"""),
    md("""## 1D pattern for a single frame

`get_1d` returns a named tuple `(q, intensity, sigma, q_unit, frames)`.
`intensity` is a 1D array for a single frame."""),
    code("""r = get_1d(scan_file, frame=spotlight)
print(f'q: {r.q.shape}   intensity: {r.intensity.shape}   unit: {r.q_unit}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r.q, r.intensity, lw=1)
if r.sigma is not None:
    ax.fill_between(r.q, r.intensity - r.sigma, r.intensity + r.sigma,
                    alpha=0.25, label='±σ')
    ax.legend()
ax.set_xlabel(f'q ({r.q_unit})')
ax.set_ylabel('Intensity')
ax.set_title(f'Frame {spotlight} — 1D')
plt.tight_layout(); plt.show()"""),
    md("""## All frames at once — waterfall

Passing `frame=None` (the default) returns every frame stacked as
`(n_frames, n_q)`."""),
    code("""allr = get_1d(scan_file)              # (n_frames, n_q)
print(f'stacked intensity: {allr.intensity.shape}')

fig, ax = plt.subplots(figsize=(7, 5))
offset = 0.0
step = np.nanmax(allr.intensity) * 0.5
for lbl, curve in zip(allr.frames, allr.intensity):
    ax.plot(allr.q, curve + offset, lw=0.8, label=f'frame {lbl}')
    offset += step
ax.set_xlabel(f'q ({allr.q_unit})')
ax.set_ylabel('Intensity (offset per frame)')
ax.set_title('1D waterfall — all frames')
if allr.intensity.shape[0] <= 10:
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""## 2D (cake / q–χ) pattern

`get_2d` returns `(q, chi, intensity, q_unit, chi_unit, frames)`.  For a
single frame `intensity` is `(n_chi, n_q)` — ready to `pcolormesh`
directly."""),
    code("""if meta['has_2d']:
    r2 = get_2d(scan_file, frame=spotlight)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(r2.q, r2.chi, r2.intensity, shading='auto', cmap='viridis')
    ax.set_xlabel(f'q ({r2.q_unit})')
    ax.set_ylabel(f'χ ({r2.chi_unit})')
    ax.set_title(f'Frame {spotlight} — 2D')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout(); plt.show()
else:
    print('No 2D data in this file (1D-only reduction).')"""),
    md("""## Thumbnail (if stored)

Some files keep a small downsampled raw image per frame."""),
    code("""try:
    thumb = get_thumbnail(scan_file, spotlight)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(thumb, cmap='gray')
    ax.set_title(f'Frame {spotlight} — thumbnail')
    ax.axis('off')
    plt.tight_layout(); plt.show()
except KeyError as e:
    print(f'No thumbnail available: {e}')"""),
    md("""## Object-style sugar: `open_scan`

If you'd rather not repeat the path, `open_scan` gives a lightweight
handle with the same readers as methods."""),
    code("""scan = open_scan(scan_file)
print(scan)                       # Scan('integrated.nxs', n_frames=N)
print('frames:', scan.frames.tolist())

first = scan.get_1d(int(scan.frames[0]))
print('first-frame 1D intensity shape:', first.intensity.shape)
# scan.get_2d(frame), scan.get_thumbnail(frame), scan.metadata also available"""),
    md("""---

### Next steps

- To **re-integrate or change reduction settings**, see notebook
  **06 — Headless Reduction Pipeline** (writes the `.nxs` these readers
  consume).
- To **fit phases / peaks** on a pattern you just read, feed `r.q` and
  `r.intensity` into notebook **03 — Phase + peak fitting**.
- For the full `xarray.Dataset` (lazy per-frame access over very large
  scans), use `read_sphere_metadata` / `read_sphere` from
  `ssrl_xrd_tools.io`.
"""),
]


if __name__ == "__main__":
    write_notebook("01_batch_integration.ipynb",        NB_BATCH_INTEGRATION)
    write_notebook("02_multigeometry_stitching.ipynb",  NB_STITCHING)
    write_notebook("03_phase_and_peak_fitting.ipynb",   NB_PHASE_PEAK)
    write_notebook("04_batch_phase_fitting.ipynb",      NB_BATCH_PHASE)
    write_notebook("05_sin2psi_analysis.ipynb",         NB_SIN2PSI)
    write_notebook("06_headless_reduction_pipeline.ipynb", NB_REDUCTION_PIPELINE)
    write_notebook("07_reading_processed_nxs.ipynb",    NB_READING_PROCESSED)
    print("\nDone.  Re-run after editing cell content above.")
