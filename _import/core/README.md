# ssrl_xrd_tools

Python tools for X-ray diffraction data processing and visualization at SSRL.

## Installation

### First time on a new machine:
```bash
git clone <repo-url>
cd ssrl_xrd_tools
conda env create -f environment.yml
conda activate ssrl_xrd_tools
```
The editable install runs automatically via environment.yml.

### Headless (no GUI):
```bash
pip install ssrl_xrd_tools
```

### With GUI (xdart):
```bash
pip install ssrl_xrd_tools[gui]
```

## Module layers

### Primitives
- `io` — image, SPEC, metadata, export
- `corrections` — detector, beam, normalization corrections
- `transforms` — unit conversions (q, tth, d, energy, hkl)

### Mid-level
- `integrate` — 1D/2D azimuthal integration, GID, batch processing
- `rsm` — reciprocal space mapping, HKL gridding

### Analysis
- `analysis.fitting` — peak models, fitting, background
- `analysis.phase` — phase matching, CIF loading
- `analysis.texture` — pole figures, ODF
- `analysis.strain` — sin2chi, d-spacing maps
- `analysis.refinement` — stub (planned: GSAS-II, FullProf)

### GUI (optional)
- `gui` — xdart interactive viewer
