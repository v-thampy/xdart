# xrd-tools

<!-- After the repo is pushed, point the badge at the real org/name:
[![PR checks](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml/badge.svg)](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml) -->

## Install

**xdart (the GUI)** — the common install:

```bash
pip install "xrd-tools[gui]"          # the xdart GUI + reduction core
uv tool install "xrd-tools[gui]"      # isolated GUI install
```

then launch with `xdart`.

**Headless core only** (no Qt anywhere, `import xrd_tools`):

```bash
pip install xrd-tools
```

Extras: `[gui]` PySide6/pyqtgraph + GUI deps, `[fitting]` pymatgen/lmfit,
`[rsm]` reciprocal-space mapping, `[dev]` test/build tooling.

SSRL X-ray diffraction toolkit: a **headless reduction core** (`xrd_tools`)
and the **xdart Qt GUI** (`xdart`) in one distribution.  Formerly the
`ssrl_xrd_tools` and `xdart` repositories — merged with full histories;
see [`MIGRATION.md`](MIGRATION.md).

### Performance: install the HDF5 stack from conda-forge

Compressed detector data — Eiger `_master.h5` files use bitshuffle+LZ4 — is
decompressed by the native HDF5 filter libraries, and that read is a large part
of processing time.  The pure-pip `h5py` / `hdf5plugin` wheels bundle a generic
(non-SIMD) filter build that decompresses Eiger frames noticeably slower
(~1.7× on Apple Silicon in our tests, e.g. a 651-frame Int-1D scan 25 s → 19 s).
For best performance, install the HDF5 stack from **conda-forge** rather than
pip:

```bash
conda install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c
```

This only affects raw-frame read speed — pyFAI integration and the writer are
unchanged.  A pure-pip install works correctly, just slower on compressed
detector data.

### Output compression (lz4 default — reading `.nxs` outside xrd-tools)

xdart writes the integrated 1D/2D stacks with **lz4+shuffle** by default (fast,
hdf5plugin filter 32004; ~gzip-class size).  **Reading those `.nxs` files requires
`hdf5plugin`** — a base dependency, so any xrd-tools/xdart environment reads them
fine.  To read them with **stock h5py elsewhere** (a collaborator's plain notebook,
a third-party tool, long-term archival) either install `hdf5plugin`, or write
portable files by setting the compression before launch:

```bash
XDART_INTEGRATED_COMPRESSION=gzip xdart   # gzip+shuffle — readable by any stock h5py
XDART_INTEGRATED_COMPRESSION=none xdart   # uncompressed
```

(Detector module gaps and decompressed values are identical either way; only the
on-disk filter changes.)

## Headless quick start

```python
from xrd_tools.reduction import (
    Integration1DPlan, Integration2DPlan, NexusSink,
    ReductionPlan, Scan, run_reduction,
)

plan = ReductionPlan(
    integration_1d=Integration1DPlan(npt=1000, unit="q_A^-1"),
    integration_2d=Integration2DPlan(npt_rad=1000, npt_azim=360),
)
scan = Scan("scan1", frames, integrator=ai)        # frames: list[ScanFrame]
run_reduction(plan, scan,
              sink=NexusSink("processed/scan1.nxs",
                             source_base="/path/to/project"))
```

The sink writes the complete, portable v2 record: integrated 1D/2D stacks,
per-frame raw-source pointers (relative to the project root), thumbnails,
scan metadata, and per-frame geometry.  Reading back:

```python
from xrd_tools.io import get_1d, get_raw_frame, open_scan, read_frame_view

scan = open_scan("processed/scan1.nxs")            # notebook sugar
q, intensity, sigma, unit, frames = get_1d("processed/scan1.nxs")
view = read_frame_view("processed/scan1.nxs", 0)   # one frame, display-ready
raw = get_raw_frame("processed/scan1.nxs", 0)      # resolves the source pointer
```

## Intensity corrections

The value in each (q, χ) cake cell or RSM voxel is only physically meaningful once per-pixel
**intensity corrections** are applied. They divide out instrumental/geometric factors so what
remains is proportional to the sample's scattering. Two groups:

**Detector / beam (apply to any integration):**

- **Solid angle.** A flat-panel pixel far from the beam centre is farther away and viewed
  obliquely, so it subtends a smaller solid angle Ω (∝ cos³2θ) and collects fewer photons for
  the same scattered intensity. Dividing by Ω converts counts → intensity-per-solid-angle and
  boosts the high-angle pixels. (pyFAI applies this by default.)
- **Polarization.** Synchrotron light is ~linearly polarized; Thomson scattering is suppressed
  along the polarization direction, imprinting a 2θ- and azimuth-dependent modulation on the
  rings. Dividing by the polarization factor removes it. (Set the degree of polarization,
  ≈ 0.99 horizontal at most beamlines.)
- **Lorentz, dark/flat/efficiency, air absorption.** Standard powder/detector corrections; Lorentz
  is a geometric weighting usually folded into 1D integration.

**Grazing incidence** (gated to GI mode; built from `xrayutilities` optics, n = 1 − δ + iβ,
critical angle αc = √2δ, with exit angle αf = arcsin(qz/k − sin αi)):

- **Footprint / illuminated area.** At grazing αi the beam spreads over a footprint ∝ 1/sin αi;
  for a sample larger than the beam the illuminated scattering volume grows with it, so measured
  intensity does too. The correction (× sin αi) normalizes it (a global scale at fixed αi;
  per-frame when αi is scanned).
- **Refraction.** The beam refracts at the surface (δ ~ 10⁻⁶); the true angles inside the film
  are αi′ = √(αi²−αc²), αf′ = √(αf²−αc²), which shift the apparent qz / ring positions near αc.
  Below αc the wave is evanescent.
- **Penetration / absorption.** Probed depth depends sharply on αi vs αc (nm below, µm above),
  and both the αi and αf paths are attenuated over the absorption length μ⁻¹. The correction
  accounts for the αi/αf-dependent path length, boosting strongly-absorbed grazing-exit pixels.
- **Fresnel transmission / Vineyard (DWBA).** |T(αi)|² and |T(αf)|² peak at αc — the bright
  **Yoneda band**. Measured intensity carries |T(αi)|²|T(αf)|²; dividing it out removes the
  Yoneda enhancement and recovers the true scattering.

Design: [`docs/design/design_intensity_corrections_jun2026.md`](docs/design/design_intensity_corrections_jun2026.md)
(per-pixel weight stack at the accumulator seam — solid-angle/polarization reuse pyFAI arrays,
the GI stack uses `xu.materials`). Interactive demo with on/off toggles, an αi slider and a
material selector: `examples/.../Stitching/Multi120_GI_Corrections_Explorer.ipynb`.

## The GUI

```bash
xdart
```

Live + batch acquisition stream through the same headless reduction spine
(parallel pyFAI workers, single writer thread, fail-loud writes).  Project
Folder mode stores raw-source paths relative to the project root, so a
processed dataset moves machines intact.

## Development

```bash
git clone <this repo> && cd xrd-tools
python -m venv .venv && . .venv/bin/activate
pip install -e ".[gui,dev,fitting,rsm]"

pytest tests/core                              # headless core suite
QT_QPA_PLATFORM=offscreen pytest tests/xdart   # GUI suite, offscreen
pytest -m display_logic                        # pure display-logic subset
```

Working notes for AI/code assistants live in [`CLAUDE.md`](CLAUDE.md);
architecture notes in `docs/`.
