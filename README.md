# xdart

<!-- After the repo is pushed, point the badge at the real org/name:
[![PR checks](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml/badge.svg)](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml) -->

**SSRL X-ray diffraction toolkit — one distribution, two import packages.**

`xdart` (version 1.0.0) is the merged successor to the former
`ssrl_xrd_tools` (the headless reduction + I/O library) and `xdart` (the
real-time Qt GUI). It ships **two import packages** from one wheel:

- **`xrd_tools`** — the headless XRD reduction + I/O core. Imports **no
  Qt / pyqtgraph**; fully usable from scripts, Jupyter notebooks, and
  automated batch pipelines at the beamline or in the lab. Built on
  [pyFAI](https://pyfai.readthedocs.io/) for azimuthal integration with added
  support for grazing incidence, multi-geometry stitching, reciprocal-space
  mapping, peak/phase/strain fitting, and a streaming reduction spine.
- **`xdart`** — the PySide6 + pyqtgraph desktop GUI for real-time and batch
  analysis. A thin consumer of `xrd_tools`.

The two former repositories were merged **with full git histories**
(`git log --follow` works across the boundary); see
[`MIGRATION.md`](https://github.com/v-thampy/xrd-tools/blob/main/MIGRATION.md).
The old `ssrl_xrd_tools` import name still
works as a **deprecation shim** that re-exports the real `xrd_tools` modules
— update imports to `xrd_tools` at your convenience.

---

## Contents

- [Install](#install)
- [Headless quick start (`xrd_tools`)](#headless-quick-start-xrd_tools)
- [Library features](#library-features)
- [Headless API guide](#headless-api-guide)
  - [Basic integration](#basic-integration)
  - [Unit conversion](#unit-conversion)
  - [Batch processing](#batch-processing)
  - [Grazing incidence](#grazing-incidence)
  - [Reciprocal space mapping](#reciprocal-space-mapping)
  - [Reading processed scan files](#reading-processed-scan-files)
  - [Peak & phase fitting](#peak--phase-fitting)
- [Intensity corrections](#intensity-corrections)
- [The GUI (`xdart`)](#the-gui-xdart)
  - [Key capabilities](#key-capabilities)
  - [GUI quick start](#gui-quick-start)
  - [Usage guide](#usage-guide)
  - [Configuration & calibration](#configuration--calibration)
  - [Troubleshooting](#troubleshooting)
- [Module architecture](#module-architecture)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Install

### Quick install — `pixi global` (recommended)

[pixi](https://pixi.sh) is a fast, self-contained package manager — no conda or
Python needed first. **Install pixi once** ([more install options](https://pixi.sh/latest/)):

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
```

```powershell
# Windows (PowerShell) — or: winget install prefix-dev.pixi
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Open a new terminal so `~/.pixi/bin` is on your PATH, then **one command** installs
xdart, puts it on your PATH, and adds a Start-menu / Applications shortcut — with
the whole fast conda-forge I/O stack:

```bash
pixi global install -c https://prefix.dev/xrd-tools -c conda-forge xdart
```

Launch with `xdart` (or the shortcut). Upgrade with `pixi global update xdart`
(or from the app: **Help → Check for Updates…**).

> The conda package is published with the release tag. Until it is live on the
> channel, use the one-line installer script below (it needs nothing preinstalled
> — not even pixi).

### One-line installer script (no conda or pixi needed)

Installs everything — Python, the fast HDF5/compression stack, and xdart — in one
step, into its own folder, without touching any existing Python or conda setup.

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/v-thampy/xrd-tools/main/scripts/install_xdart.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/v-thampy/xrd-tools/main/scripts/install_xdart.ps1 | iex"
```

- **Needs nothing preinstalled** — no conda, no Python. It bootstraps a
  self-contained [pixi](https://pixi.sh) workspace under `~/.local/share/xdart`
  (`%LOCALAPPDATA%\xdart` on Windows) and never edits your shell config or an
  existing conda install.
- **Fast by construction** — it uses the conda-forge builds of the HDF5/compression
  stack (the fastest Eiger bitshuffle/LZ4 decode) plus `xdart[gui]` from PyPI,
  resolved in one solve with a lockfile. This is the same layering the manual
  conda steps below do — only assembled for you.
- **Launch** with `xdart`. **Upgrade** by re-running the same line.
- **Extras**: set `XDART_EXTRAS` before the command, e.g.
  `XDART_EXTRAS="gui,fitting" curl -fsSL … | bash`.
- If `xdart` launches an old version, run `hash -r` (or open a new terminal) — the
  installer prints the specifics when it detects a shadowing install.

### Using conda / mamba

The classic path: a **fresh conda environment** with the conda-forge compiled I/O
stack, plus `xdart` from PyPI. No conda yet? Install one first (pick either):

- **[Miniforge](https://github.com/conda-forge/miniforge)** — recommended;
  conda-forge by default, smaller, ships `mamba`.
- **[Miniconda](https://www.anaconda.com/download/success)** — Anaconda's minimal
  distribution.

On Windows, open the "Miniforge Prompt" / "Anaconda Prompt" the installer created;
on macOS / Linux, open a regular terminal (the installer wires `conda`/`mamba` into
your shell).

```bash
# 1. create + activate a fresh env (use `conda` in place of `mamba` if you prefer —
#    mamba is just the faster solver)
mamba create -n xrd python=3.13 -y
mamba activate xrd

# 2. the conda-forge fast I/O stack, then the xdart GUI from PyPI
mamba install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c
pip install "xdart[gui]"
```

then launch with `xdart`. (Already have an environment? Just run step 2 in it.)

### Using pip / uv

Requires **Python ≥ 3.11**. `xdart` is a normal PyPI package and installs
anywhere:

```bash
pip install "xdart[gui]"          # the xdart GUI + reduction core
uv tool install "xdart[gui]"      # isolated GUI install
```

then launch with `xdart`. Note the Eiger bitshuffle/LZ4 decode is measurably slower
with the pure-pip HDF5 wheels than the conda-forge builds
([see below](#performance-install-the-hdf5-stack-from-conda-forge)), and
lz4-compressed outputs need `hdf5plugin` (a base dep) to read outside xdart.

**Headless core only** (no Qt anywhere, `import xrd_tools`):

```bash
pip install xdart
```

> **Upgrading from the old `xdart` / `ssrl_xrd_tools`?** Uninstall the legacy
> packages first so their entry points and shims don't shadow `xdart`:
>
> ```bash
> pip uninstall -y xdart ssrl_xrd_tools
> ```
>
> Then install `xdart` as above. See
> [`MIGRATION.md`](https://github.com/v-thampy/xrd-tools/blob/main/MIGRATION.md)
> for the full import-name migration.

### Headless / notebooks with pixi

For notebook analysis or batch scripts, a pixi workspace gives you the same fast
stack **plus a lockfile** that makes the environment reproducible — a drop-in
replacement for a per-project conda/mamba env. Existing conda envs keep working;
this is an option, not a migration.

```bash
mkdir my-analysis && cd my-analysis
pixi init
pixi add python=3.13 h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c jupyterlab
pixi add --pypi "xdart[notebook,fitting]"
pixi run jupyter lab
```

- The env lives in `./.pixi/` next to the notebooks; commit `pixi.toml` +
  `pixi.lock` and anyone (including future-you) reproduces the exact env with one
  `pixi install`.
- Add `[rsm]` via conda where possible — `pixi add xrayutilities` (conda-forge)
  avoids the missing macOS-arm64 PyPI wheels.
- `pixi run python script.py` runs a headless batch script; `pixi shell` in the
  workspace is the equivalent of `conda activate`.

**Shared beamline environment (VS Code).** A pixi env is a normal prefix
(`<workspace>/.pixi/envs/default/bin/python`), so the one-shared-env /
many-user-directories pattern maps 1:1. Put **one** pixi workspace at a shared path
(e.g. `/shared/xrd-env/` — its `pixi.toml`, `pixi.lock`, and `.pixi/`); users open
their own notebook folders in VS Code and select that env's `bin/python` as the
interpreter/kernel (VS Code auto-discovers pixi envs; "Enter interpreter path"
always works). Register it by name once so it appears in every kernel picker:

```bash
cd /shared/xrd-env && pixi run python -m ipykernel install --prefix /usr/local \
    --name xdart --display-name "XRD Tools (shared)"
```

Admins update the shared env with `pixi update` in that directory; the lockfile
rebuilds it identically on a new machine (`pixi install`).

### Extras

The base install is **headless / scriptable** (`core`, `io`, `integrate`,
`viz`). Domain-specific features live behind [PEP 621
extras](https://peps.python.org/pep-0621/) so the dependency footprint stays
modest for batch / pipeline / CI use:

| Extra        | What it enables                                          | Packages                                                    |
| ------------ | -------------------------------------------------------- | ----------------------------------------------------------- |
| *(base)*     | `core`, `io`, `integrate`, `viz` — headless / batch      | numpy, scipy, pandas, xarray, h5py, hdf5plugin, nexusformat, fabio, silx, pyFAI, pyyaml, joblib, natsort, matplotlib, plotly |
| `[gui]`      | the `xdart` desktop GUI **+ its analysis tools** (bundles `[fitting]` + `[rsm]`) | PySide6, pyqtgraph, qtawesome, imagecodecs, imageio, lmfit, pymatgen, xrayutilities, pyevtk |
| `[fitting]`  | `analysis.fitting.*` — peak / phase / strain fitting     | lmfit, pymatgen                                             |
| `[rsm]`      | `rsm.*` — reciprocal-space mapping, VTK export           | xrayutilities, pyevtk                                       |
| `[notebook]` | self-contained Jupyter environment                       | ipywidgets, anywidget, ipyfilechooser, ipykernel, ipympl, jupyterlab |
| `[all]`      | everything except dev                                    | `xdart[fitting,rsm,gui,notebook]`                       |
| `[dev]`      | test / build / release tooling                           | pytest, pytest-timeout, build, twine, tifffile              |

Extras compose. `[gui]` already bundles `[fitting]` + `[rsm]` — the GUI surfaces
Peak/Phase Fitting and the Grazing/GI/RSM workflow — so `pip install "xdart[gui]"`
(and the conda package) give you the **complete** GUI with no missing-dependency prompts.

> **Tip — use [`uv`](https://docs.astral.sh/uv/) if you have it.** It is a
> drop-in pip replacement that is typically 10–100× faster on cold installs.
> With the scientific-stack dependency tree (pyFAI, h5py, silx, PySide6, …)
> that is often the gap between a fresh-env install finishing in a few
> seconds vs. several minutes. `pip install uv` (or `brew install uv` /
> `winget install astral-sh.uv`), then prefix the commands with `uv `.

### Performance: install the HDF5 stack from conda-forge

Compressed detector data — Eiger `_master.h5` files use bitshuffle+LZ4 — is
decompressed by the native HDF5 filter libraries, and that read is a large part
of processing time. The pure-pip `h5py` / `hdf5plugin` wheels bundle a generic
(non-SIMD) filter build that decompresses Eiger frames noticeably slower
(~1.7× on Apple Silicon in our tests, e.g. a 651-frame Int-1D scan 25 s → 19 s).
For best performance, install the HDF5 stack from **conda-forge** rather than
pip:

```bash
conda install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c
```

This only affects raw-frame read speed — pyFAI integration and the writer are
unchanged. A pure-pip install works correctly, just slower on compressed
detector data.

### Output compression (lz4 default — reading `.nxs` outside xdart)

xdart writes the integrated 1D/2D stacks with **lz4+shuffle** by default (fast,
hdf5plugin filter 32004; ~gzip-class size). **Reading those `.nxs` files requires
`hdf5plugin`** — a base dependency, so any xdart environment reads them
fine. To read them with **stock h5py elsewhere** (a collaborator's plain notebook,
a third-party tool, long-term archival) either install `hdf5plugin`, or write
portable files by setting the compression before launch:

```bash
XDART_INTEGRATED_COMPRESSION=gzip xdart   # gzip+shuffle — readable by any stock h5py
XDART_INTEGRATED_COMPRESSION=none xdart   # uncompressed
```

(Detector module gaps and decompressed values are identical either way; only the
on-disk filter changes.)

---

## Headless quick start (`xrd_tools`)

The canonical reduction path — the one the GUI itself drives — is the
streaming reduction spine: choose a `ReductionPlan`, supply a `Scan`, point it
at a sink.

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
scan metadata, and per-frame geometry. Reading back:

```python
from xrd_tools.io import get_1d, get_raw_frame, open_scan, read_frame_view

scan = open_scan("processed/scan1.nxs")            # notebook sugar
q, intensity, sigma, unit, frames = get_1d("processed/scan1.nxs")
view = read_frame_view("processed/scan1.nxs", 0)   # one frame, display-ready
raw = get_raw_frame("processed/scan1.nxs", 0)      # resolves the source pointer
```

---

## Library features

- **1D/2D azimuthal integration** — fast azimuthal integration via pyFAI with
  full detector-geometry support.
- **Grazing-incidence diffraction (GID)** — specialized integration for
  surface-sensitive and thin-film measurements (pyFAI `FiberIntegrator`).
- **Multi-geometry stitching** — combine multiple detector angles into a
  seamless extended-Q 1D/2D pattern.
- **Reciprocal-space mapping (RSM)** — build 3D reciprocal-space volumes with
  HKL gridding (`[rsm]`).
- **Streaming reduction spine** — parallel pyFAI workers + a single writer
  thread, bounded in-flight memory, **fail-loud** `finish()` (a failed write
  raises, never silently succeeds).
- **NeXus/HDF5 I/O** — standards-compliant, schema-strict stacked v2 records;
  full NumPy/pandas object serialization via the HDF5 codec.
- **Portable raw-source paths** — Project-Folder mode stores raw pointers
  relative to the project root, so a processed dataset moves machines intact.
- **SPEC file parsing** — scan commands, geometries, counter data, metadata.
- **Peak fitting** — lmfit-based single- and multi-peak fitting with
  selectable backgrounds (linear, constant, Chebyshev, polynomial, SNIP)
  (`[fitting]`).
- **Phase pattern fitting** — multi-phase pseudo-Voigt fitting with
  pymatgen-derived peak positions and template intensities (`PhaseFitter`)
  (`[fitting]`).
- **Strain analysis** — sin²(ψ) method for biaxial stress/strain, with
  optional (E, ν) inputs for direct stress output (`[fitting]`).
- **Batch processing** — directory watching and automated pipeline execution.

> **Roadmap (not yet implemented):** texture analysis (pole figures, ODF),
> Rietveld/LeBail refinement, automated phase matching against structure
> databases, additional standalone correction modules. The corresponding
> `analysis/texture.py`, `analysis/refinement.py` entries are placeholders
> reserved for these features. Likewise `xrd_tools.gui.main` is a reserved
> standalone-launcher entry point (raises `NotImplementedError`) — use `xdart`.

---

## Headless API guide

> All examples import from `xrd_tools`. The legacy `ssrl_xrd_tools` import
> name still resolves (deprecation shim) but is not recommended for new code.

### Basic integration

`integrate_1d` / `integrate_2d` take a detector image **and a configured
pyFAI integrator** (not a PONI directly — build the integrator from a PONI
with `poni_to_integrator`).

```python
from xrd_tools.io import read_image, write_nexus
from xrd_tools.integrate import load_poni, poni_to_integrator, integrate_1d

# Load calibration and build a pyFAI integrator
poni = load_poni("path/to/calibration.poni")
ai = poni_to_integrator(poni)

# Read a detector image
img = read_image("path/to/image.h5")

# Azimuthal integration -> IntegrationResult1D (.radial, .intensity, .sigma, .unit)
result_1d = integrate_1d(img, ai, npt=1000, unit="q_A^-1")

# Save to NeXus
write_nexus("output.h5", result_1d)
```

### Unit conversion

Energy is passed as `energy_keV`. d-spacing helpers are `q_to_d` / `d_to_q`
(there is no `tth_to_dspacing` — convert via `tth_to_q` then `q_to_d`).

```python
from xrd_tools.transforms import tth_to_q, q_to_d, d_to_q

q = tth_to_q(two_theta, energy_keV=15.0)   # 2theta (deg) -> Q (A^-1)
d = q_to_d(q)                              # Q (A^-1) -> d-spacing (A)
q = d_to_q(d)                              # d-spacing (A) -> Q (A^-1)
```

### Batch processing

```python
from xrd_tools.integrate import process_series

# Process a sequence of scans with one configured integrator;
# each writes <output_dir>/<scan_stem>_processed.h5
outputs = process_series(
    scan_paths=["./raw_data/scan1", "./raw_data/scan2"],
    ai=ai,
    output_dir="./processed",
)
```

For directory-watched live ingestion see `integrate.DirectoryWatcher`.

### Grazing incidence

GI integration uses a pyFAI `FiberIntegrator`, built from a PONI plus the
grazing-incidence geometry, then integrated with an `incident_angle`.

```python
from xrd_tools.integrate import create_fiber_integrator, integrate_gi_1d

fi = create_fiber_integrator(poni, incident_angle=0.2)   # degrees
result_gi = integrate_gi_1d(img, fi, unit="qoop_A^-1", incident_angle=0.2)
```

`integrate_gi_2d`, `integrate_gi_polar_1d`, and `integrate_gi_exitangles_1d`
cover the 2D cake, polar, and exit-angle modes respectively.

### Reciprocal space mapping

```python
from xrd_tools.rsm import ExperimentConfig, RSMVolume

config = ExperimentConfig(...)     # crystal structure + Q-space geometry
volume = RSMVolume(...)            # build a 3D reciprocal-space grid
rsm_data = volume.to_hkl_grid()    # map to HKL
```

(`ExperimentConfig` / `RSMVolume` constructor signatures are illustrative —
see `xrd_tools.rsm` for the current parameters. Requires the `[rsm]` extra.)

### Reading processed scan files

Once a scan has been reduced (by xdart or the headless pipeline) the results
live in a v2 NeXus `.nxs` file. The `get_*` convenience readers pull 1D / 2D
patterns, thumbnails, and metadata back out in one line — no xarray knowledge
required. A processed file is a **scan**; each reader takes a **frame** index
(or `None` for all frames).

```python
from xrd_tools.io import (
    get_frames, get_metadata, get_1d, get_2d, get_thumbnail, open_scan,
)

get_frames("scan.nxs")              # frame labels
meta = get_metadata("scan.nxs")     # sample, energy, wavelength, axes, motors
q, intensity, sigma, unit, frames = get_1d("scan.nxs")    # all frames
cake = get_2d("scan.nxs", 2)        # frame 2 — (chi, q) oriented

# object-style sugar
scan = open_scan("scan.nxs")
scan.frames, len(scan), scan.get_1d(2), scan.metadata
```

For per-frame, display-ready reconstruction (the reload half of the
live≡batch≡reload equivalence spine) use `read_frame_view` /
`read_frame_views`.

### Peak & phase fitting

```python
from xrd_tools.analysis.phase import PhaseModel
from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

# Build a phase from a CIF and compute its peaks
phase = PhaseModel.from_cif("Fe.cif", name="alpha-Fe")
phase.calculate_peaks(q_range=(1.0, 7.0))

# Fit a measured 1D pattern with one or more phases
fitter = PhaseFitter(q, intensity, background="snip")
fitter.add_phase(phase)
result = fitter.fit(fitter.build_params())
```

Structure-agnostic single-/multi-peak fitting lives under
`xrd_tools.analysis.fitting` (lmfit-based, selectable backgrounds). Requires
the `[fitting]` extra.

---

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

Implemented as a per-pixel weight stack at the accumulator seam — solid-angle/polarization
reuse pyFAI arrays, the GI stack uses `xu.materials`. Interactive demo with on/off toggles, an αi slider and a
material selector: `examples/notebooks/02_multigeometry_stitching.ipynb`.

---

## The GUI (`xdart`)

```bash
xdart
```

xdart is a pyFAI-based desktop GUI for real-time azimuthal integration and
visualization of synchrotron X-ray diffraction data, built with PySide6 and
pyqtgraph for high-performance interactive plotting. Live **and** batch
acquisition stream through the **same** headless reduction spine (parallel
pyFAI workers, single writer thread, fail-loud writes); the GUI reloads from
the `.nxs` at end-of-batch.

### Key capabilities

- **Real-time 1D/2D azimuthal integration** using pyFAI.
- **Batch processing** of image series with parallel multicore support.
- **Grazing-incidence diffraction (GID)** integration using pyFAI
  `FiberIntegrator`.
- **Live monitoring** of ongoing experiments with directory-watched file
  ingestion.
- **NeXus/HDF5 data format** with full metadata preservation; portable
  Project-Folder relative-path storage.
- **Interactive 2D detector-image visualization** with zoom, pan, and masking.
- **Unit conversion** between Q (Å⁻¹) and 2θ (°) with wavelength awareness.
- **Background subtraction** (single file, series average, or
  directory-matched).
- **Masking tools** for bad pixels, beamstop shadows, and detector edges.
- **Calibration management** via PONI files with visual feedback.
- **Raw-image preview thumbnails** for quick file browsing.
- **Automatic metadata discovery** — the Meta Type selector defaults to `auto`:
  per-image sidecar metadata (`.txt`, `.pdi`, QXRD-style
  `image.tif.metadata`, and other structured name=value sidecars) is found and
  parsed automatically; choose `none` to disable or `spec` for SPEC files.
- **Overlay / Waterfall comparison across scans** — overlay 1D patterns and
  slice cuts from multiple frames, and **pin** the current slice cut (Cmd+P)
  to keep it on the plot for comparison; the overlay survives compatible scan
  boundaries, so cuts from successive scans (e.g. a chi-texture series) can be
  compared directly.

### GUI quick start

```bash
xdart          # launch the GUI
```

Or from Python:

```python
from xdart.xdart_main import main
main()
```

**Basic workflow:**

1. **Launch xdart** and wait for the main window to open.
2. **Set calibration**: browse and select your PONI calibration file in the
   right panel.
3. **Select data**: choose an image file or directory containing images.
4. **Choose processing mode**: Batch 1D, Live 1D, Batch 2D, Live 2D, or Viewer.
5. **Configure parameters** (optional): background subtraction, masking, or
   advanced integration options.
6. **Click Start**: processing begins; monitor progress and view results in
   real time.

### Usage guide

#### Batch mode (Batch 1D, Batch 2D)

Process all frames in a selected image file or directory at once. Ideal for
complete datasets acquired in a single experiment.

1. Select your PONI file.
2. Select your image file or directory.
3. Choose "Batch 1D" or "Batch 2D" from the mode dropdown.
4. Adjust processing parameters as needed.
5. Click Start.

Results are saved as NeXus HDF5 files (`.nxs`) in the output directory. Batch
mode is a performance switch: it runs silently (no per-frame display refresh),
then the GUI reloads from the `.nxs` at end-of-batch.

#### Append vs Replace

The write-mode toggle (Cmd+Shift+A) controls whether a run **appends** new
frames to the existing processed `.nxs` or **replaces** it. Re-running Append
on an already-processed scan is near-instant: frames already in the output are
skipped without re-reading the raw data. If the current processing
configuration no longer matches the existing scan (e.g. you switched Standard
<-> Grazing), Run shows a **Replace existing integration?** dialog —
click **Yes** to switch to Replace and re-process all frames under the new
configuration, or **No** to leave the existing scan untouched (the run does
not start).

#### Live mode (Live 1D, Live 2D)

Monitor a directory for new image files and integrate them as they arrive —
real-time feedback during an active beamline experiment.

1. Set the PONI file and image directory.
2. Choose "Live 1D" or "Live 2D".
3. Click Start.
4. xdart watches the directory and processes new files automatically.

Live and Batch are pure mode toggles; the single **Start** button morphs
green → orange (Pause / Resume).

#### Keyboard shortcuts

All shortcuts are also discoverable in the File and Run menus (Cmd on macOS,
Ctrl on Linux/Windows):

- **Cmd+R** — Run / Pause / Resume the current processing run.
- **Cmd+Shift+C** — Stop the run.
- **Cmd+Shift+A** — toggle the write mode between Append and Replace.
- **Cmd+P** — pin the current slice cut (adds it to the Overlay).
- **Cmd+O** / **Cmd+S** — load / save the xdart settings (Config).

#### Viewer mode

Browse and display previously saved NeXus files without re-integrating.

#### Integration axes and units

The 1D integration panel's axis dropdown offers:

- **Q (Å⁻¹)**: scattering-vector magnitude; independent of wavelength.
- **2θ (°)**: scattering angle; depends on the wavelength in your PONI file.

The 2D integration panel offers:

- **Q-χ**: radial-azimuthal in reciprocal space.
- **2θ-χ**: radial-azimuthal in angle space.

Unit conversion respects your calibration file's wavelength automatically.

#### Grazing-incidence diffraction (GID)

For surface-sensitive measurements, xdart supports grazing-incidence geometry
using pyFAI's `FiberIntegrator`:

1. In the **Grazing Incidence** section of the parameter tree, enable GI mode.
2. Specify the incident angle and sample-normal direction (if using advanced
   geometry).
3. The integrator panel switches to GI-specific modes:
   - **1D modes**: Qip (in-plane), Qoop (out-of-plane), Q-total.
   - **2D modes**: Qip-Qoop, Q-χ.
4. Process as normal; output reflects the rotated reciprocal-space axes.

#### Advanced integration settings

Click **"Advanced..."** next to the processing mode dropdown for detailed pyFAI
parameters:

- **Solid-angle correction**: account for detector solid-angle variations.
- **Dummy values**: mark pixels to ignore in integration.
- **Polarization factor**: apply polarization correction for synchrotron
  radiation.
- **Integration method**: choose the algorithm (e.g. histogram, csr,
  full-split).
- **Radial range**: manually clip the Q or 2θ range (overrides auto-detection).
- **Azimuthal range**: select only certain χ sectors.

> **Auto Mask Saturated** (the "Auto" toggle in the pixel-rejection row,
> default **ON**) is the authoritative toggle for saturated-pixel masking,
> applied identically across live, batch, and reintegrate. ON masks negatives,
> the uint32 dead/hot sentinel (Eiger), and the integer-dtype saturation
> ceiling — but the ceiling mask only fires when a whole block of the frame
> (>=1e-4 of pixels) sits at the ceiling, so a handful of legitimately
> saturated strong Bragg pixels are kept. OFF masks nothing — the raw frame,
> including the uint32-max sentinel, passes straight through to the integration.

#### Background subtraction

Configure in the **Background** section of the parameter tree:

- **No background**: raw data only.
- **Single file**: subtract a single dark image.
- **Series average**: average all images in a background directory, then
  subtract.
- **Directory-matched**: match each sample image to a background image by
  filename pattern.

Background frames are integrated with the same parameters as sample frames for
consistency.

#### Calibration and masking

The integrator panel includes **Calibrate** and **Make Mask** buttons:

- **Calibrate** launches the pyFAI-calib2 module for interactive detector
  calibration. Use a calibration standard (e.g. LaB6, CeO2) to refine detector
  geometry and generate a PONI file.
- **Make Mask** launches the pyFAI mask-drawing tool to interactively draw
  regions on a detector image and create/edit a bad-pixel mask.

Additional masking options in the parameter tree:

- **Preset masks**: load a mask file from disk.
- **Threshold masking**: automatically mask pixels above/below intensity
  thresholds.
- **Detector masks**: apply detector-specific dead-pixel maps.

Masks are saved with your results in the NeXus file for reproducibility.

#### Data export and saving

- **Automatic export**: processed 1D/2D data is saved as NeXus HDF5 (`.nxs`)
  during batch processing.
- **Manual export**: use the **Save** button in the display area to export the
  currently displayed pattern.
- **Metadata**: all integration parameters, calibration info, and masking are
  stored in the HDF5 structure.

### Configuration & calibration

#### PONI files

xdart uses pyFAI PONI (PyFAI Object Containing Necessary Information) files for
detector calibration. A PONI file contains:

- detector geometry (pixel size, shape, name),
- incident-beam center location,
- sample-to-detector distance,
- wavelength,
- detector rotation (if any).

Generate a PONI file with the **Calibrate** button in xdart's integrator panel
(which launches pyFAI-calib2), or from the command line:

```bash
pyFAI-calib2
```

Refer to the pyFAI documentation for detailed calibration procedures.

### Troubleshooting

**pyFAI installation fails on macOS/Windows:**
Install via conda instead of pip; conda packages include pre-built binaries.

```bash
conda install -c conda-forge pyfai
```

**GUI doesn't appear or crashes on startup:**
Check that PySide6 is installed and your Qt plugins are accessible:

```bash
python -c "from PySide6 import QtWidgets; print('PySide6 OK')"
```

**Slow integration or freezing:**
Ensure multicore processing is enabled, and reduce the radial resolution if
working with very large detectors.

**PONI file not recognized:**
Verify the PONI file is text-based `key=value` pairs and that paths are
absolute or relative to the working directory.

**Re-Integrate memory use (sizing a low-RAM machine):**
The 2D display/staging window is RAM-aware: xdart keeps up to
`clamp(0.25 x total_RAM / frame_size, 16, 64)` frames of heavy data in memory
(64 on a large-RAM machine; 16 on a small one), logged at run start as
`heavy window: N frames (...)`. Set `XDART_HEAVY_WINDOW=<n>` (8-128) to pin
it. A Re-Integrate pass adds a transient raw-frame peak of about
`2 x (cores) x ~18 MB`, bounded by worker cores, not scan length; on a
low-RAM machine lower the core count and/or set a smaller
`XDART_HEAVY_WINDOW`.

**Environment variables:**
`XDART_FLUSH_MS` (default 150, floor 110) controls the heavy image-update
quantum during live runs; `XDART_LIST_MS` (default 60) controls the fast
frame-list/cursor refresh; `XDART_PERF=1` logs per-flush display-pipeline
timings; `XDART_HEAVY_WINDOW=<n>` (8-128) pins the RAM-aware in-memory
heavy-frame window.

---

## Module architecture

One distribution, two import packages under `src/`. Anything that does not
need Qt belongs in `xrd_tools` ("keep xdart thin").

### `xrd_tools` — headless core (no Qt)

- **`core/`** — pure, import-light data contracts (no Qt/h5py/fabio/pyFAI at
  import). `containers` (`PONI`, `IntegrationResult1D/2D`), `frame_view`
  (`Axis`, `TwoDKind`, `FrameView`, the GI-kind classifier), `scan`
  (`ScanFrame` / `Scan` / `FrameSource` — the reduction-input contracts),
  `filters`, `geometry/`, `metadata`, `hdf5` (universal NumPy/pandas/Python
  codec, lazily re-exported), `provenance`, `config`.
- **`io/`** — persistence + readers. `schema.py` (schema-as-code), `nexus.py`
  (stacked v2 writer/reader + strict validators), `nexus_record.py` (per-frame
  record primitives + thumbnails), `read.py` (`get_1d/2d/thumbnail/metadata`,
  `get_raw_frame`, `open_scan` / `ProcessedScan`, portable-path resolution),
  `frame_view.py` (`read_frame_view` / `read_frame_views`), `image.py`,
  `image_source.py`, `spec.py`, `metadata.py`, `nexus_inspect.py`, `export.py`,
  `tiled.py`.
- **`sources/`** — source-readiness and capability contracts, including
  `describe_source_readiness`, shared by headless callers and xdart run gating.
- **`reduction/`** — the streaming spine. `ReductionSession` (parallel workers
  + single writer thread, bounded in-flight, fail-loud `finish()`),
  `run_reduction`, sinks (`NexusSink`, `XYESink`, `MemorySink`,
  `CompositeSink`), GI freeze policies, `FlushPolicy`.
- **`session/`** — headless session/display contracts: readiness, display
  decision logic, frame records, publication projections, and shared staging
  budgets used by the GUI.
- **`integrate/`** — pyFAI integration + GI (`integrate_1d/2d`,
  `create_fiber_integrator`, `integrate_gi_*`, `stitch_1d/2d`,
  calibration: `load_poni` / `poni_to_integrator` / `poni_to_fiber_integrator`),
  batch (`process_series`, `DirectoryWatcher`).
- **`transforms/`** — unit conversions (`tth_to_q`, `q_to_tth`, `q_to_d`,
  `d_to_q`, `energy_to_wavelength`) and angular calculations.
- **`rsm/`** — reciprocal-space mapping (`ExperimentConfig`, `RSMVolume`,
  HKL gridding, VTK export).
- **`analysis/`** — `fitting` (lmfit peak + `PhaseFitter` phase fitting,
  backgrounds incl. SNIP), `phase` (`PhaseModel` over pymatgen), `strain`
  (sin²ψ).
- **`corrections/`** — intensity-correction modules (see above).
- **`viz/`** — matplotlib/plotly headless plotting (no Qt).
- **`gui/`** — Jupyter-widget viewers for notebooks (`powder_1d_viewer`,
  `powder_2d_viewer`, `rsm_viewer`, `napari_viewer`). `gui.main` is a reserved
  entry point only — it raises `NotImplementedError`; the desktop GUI is
  `xdart`.

### `xdart` — Qt GUI (thin consumer)

- **`xdart_main.py`** — thin Qt-probing entry (the `xdart` console script;
  `xdart.xdart_main:main`).
- **`modules/`** — `reduction.py` (the LiveScan→core adapter),
  `frame_publication.py` (the Qt-free GUI envelope: `FramePublication`,
  `PublicationStore`, `validate_publication`), `ewald/` (`LiveScan` /
  `LiveFrame`, the GUI NeXus writer).
- **`gui/tabs/static_scan/`** — the display layer: a pure, Qt-free decision
  core (`display_logic.py`) feeds a thin generation-stamped renderer
  (`display_frame_widget.py`) via one controller per mode
  (`display_controllers.py`). Viewer controllers resolve files through the
  headless `xrd_tools.io` APIs — xdart never opens HDF5 to guess.

Layer map: [`docs/ARCHITECTURE.md`](https://github.com/v-thampy/xrd-tools/blob/main/docs/ARCHITECTURE.md).

---

## Development

```bash
git clone https://github.com/v-thampy/xrd-tools.git && cd xrd-tools
python -m venv .venv && . .venv/bin/activate
pip install -e ".[gui,dev,fitting,rsm]"

pytest tests/core                              # headless core suite
QT_QPA_PLATFORM=offscreen pytest tests/xdart   # GUI suite, offscreen
pytest -m display_logic                        # pure display-logic subset
pytest -m "not slow"                           # skip full-size detector cases
```

---

## Contributing

Contributions are welcome. Fork the repository, create a feature branch
(`git checkout -b my-feature`), make your changes with tests, and submit a
pull request with a clear description. For bug reports or feature requests,
open an issue on the GitHub repository.

## License

First-party code is released under the **MIT License** (see
[LICENSE](https://github.com/v-thampy/xrd-tools/blob/main/LICENSE)); code
inherited from `ssrl_xrd_tools` is BSD-3-Clause (see
`licenses/LICENSE-ssrl_xrd_tools`). SPDX: `MIT AND BSD-3-Clause`.

## Citation

If you use `xdart` (`xrd_tools` / `xdart`) in your research, please cite:

```
xdart: SSRL X-ray diffraction toolkit (headless reduction core + xdart GUI)
https://github.com/v-thampy/xrd-tools
```

(Formal publication citation coming soon.)

## Acknowledgments

Developed at the [Stanford Synchrotron Radiation Lightsource
(SSRL)](https://www-ssrl.slac.stanford.edu/), SLAC National Accelerator
Laboratory. The project builds on the excellent `pyFAI` library and benefits
from the broader scientific Python ecosystem including NumPy, SciPy, lmfit,
xrayutilities, and pymatgen. Grateful thanks to the pyFAI community and to all
collaborators and users who provide feedback and improvements.

## Contact

For questions, feedback, or collaboration inquiries:

**Vivek Thampy** — vthampy@stanford.edu
