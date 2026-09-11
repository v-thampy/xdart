# Installation details

For the recommended Pixi global installation and updates, see the
[README](../README.md#quick-start). This guide covers alternative methods, notebook
environments, editable source installs, and dependency details.

- [One-line installer](#one-line-installer-script-no-conda-or-pixi-needed)
- [Updating an installer workspace](#updating-an-installer-workspace)
- [Conda / mamba](#using-conda--mamba)
- [pip / uv](#using-pip--uv)
- [Notebook environments](#headless--notebooks-with-pixi)
- [Editable Pixi installs, including Windows](#editable-install-with-pixi-recommended)
- [Extras](#extras)
- [HDF5 and compression performance](#performance-install-the-hdf5-stack-from-conda-forge)
- [Reading compressed output](#output-compression-lz4-default--reading-nexus-outside-xdart)
- [Tests and packaging](#tests-and-packaging)
- [Published releases and updates](#published-releases-and-updates)

## One-line installer script (no conda or pixi needed)

Installs everything — Python, the fast HDF5/compression stack, and xdart — in one
step, into its own folder, without touching any existing Python or conda setup.

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/v-thampy/xdart/main/scripts/install_xdart.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/v-thampy/xdart/main/scripts/install_xdart.ps1 | iex"
```

- **Needs nothing preinstalled** — no conda, no Python. It bootstraps a
  self-contained [pixi](https://pixi.sh) workspace under `~/.local/share/xdart`
  (`%LOCALAPPDATA%\xdart` on Windows) and never edits your shell config or an
  existing conda install.
- **Fast by construction** — it uses the conda-forge builds of the HDF5/compression
  stack (the fastest Eiger bitshuffle/LZ4 decode) plus `xdart[gui]` from PyPI,
  resolved in one solve with a lockfile. This is the same layering the manual
  conda steps below do — only assembled for you.
- **Launch** with `xdart`. **Upgrade** using the [private-workspace instructions below](#updating-an-installer-workspace).
- **Extras**: set `XDART_EXTRAS` before the command, e.g.
  `curl -fsSL … | XDART_EXTRAS="gui,notebook" bash` on macOS/Linux.
- If `xdart` launches an old version, run `hash -r` (or open a new terminal) — the
  installer prints the specifics when it detects a shadowing install.

## Updating an installer workspace

The **one-line installer** creates a private Pixi workspace, so it is updated
separately from global environments. Use **Help → Check for Updates…**, or close
xdart and update the workspace directly (default installation paths shown):

```bash
# macOS / Linux
~/.local/share/xdart/pixi/bin/pixi update --manifest-path ~/.local/share/xdart/pixi.toml
```

```powershell
# Windows (PowerShell)
& "$env:LOCALAPPDATA\xdart\pixi\bin\pixi.exe" update --manifest-path "$env:LOCALAPPDATA\xdart\pixi.toml"
```

## Using conda / mamba

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

## Using pip / uv

Requires **Python ≥ 3.13**. Python **3.13** is the default used by the Pixi
workspaces, installer scripts, and CI. `xdart` is a normal PyPI package:

```bash
pip install "xdart[gui]"          # the xdart GUI + reduction core
uv tool install "xdart[gui]"      # isolated GUI install
```

then launch with `xdart`. Note the Eiger bitshuffle/LZ4 decode is measurably slower
with the pure-pip HDF5 wheels than the conda-forge builds
([see below](#performance-install-the-hdf5-stack-from-conda-forge)), and
lz4-compressed outputs need `hdf5plugin` (a base dep) to read outside xdart.

The GUI installation includes **[NeXpy](https://nexpy.github.io/nexpy/)** for opening
the selected processed NeXus file and
**[silx view](https://silx.readthedocs.io/en/stable/applications/view.html)** for
general HDF5 browsing. Both run in separate
windows using the same installed environment; no extra viewer environment is
needed. Use the buttons below the Data Browser. Advanced executable overrides
are `XDART_NEXPY_EXECUTABLE` and `XDART_SILX_EXECUTABLE` (absolute executable
paths, not shell commands).

**Headless core only** (no Qt anywhere, `import xrd_tools`):

```bash
pip install xdart
```

## Headless / notebooks with pixi

For notebook analysis or batch scripts, a pixi workspace gives you the same fast
stack **plus a lockfile** that makes the environment reproducible — a drop-in
replacement for a per-project conda/mamba env. Existing conda envs keep working;
this is an option, not a migration.

```bash
mkdir my-analysis && cd my-analysis
pixi init -c conda-forge
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

## Extras

The base install is **headless / scriptable** (`core`, `io`, `integrate`,
`viz`). Domain-specific features live behind [PEP 621
extras](https://peps.python.org/pep-0621/) so the dependency footprint stays
modest for batch / pipeline / CI use:

| Extra        | What it enables                                          | Packages                                                    |
| ------------ | -------------------------------------------------------- | ----------------------------------------------------------- |
| *(base)*     | `core`, `io`, `integrate`, `viz` — headless / batch      | numpy, scipy, pandas, xarray, h5py, hdf5plugin, nexusformat, fabio, silx, pyFAI, pyyaml, joblib, natsort, matplotlib, plotly |
| `[gui]`      | the `xdart` desktop GUI **+ its analysis tools and external viewers** (bundles `[fitting]` + `[rsm]`) | PySide6, pyqtgraph, qtawesome, imagecodecs, imageio, lmfit, pymatgen, xrayutilities, pyevtk, nexpy (silx already in base) |
| `[fitting]`  | `analysis.fitting.*` — peak / phase / strain fitting     | lmfit, pymatgen                                             |
| `[rsm]`      | `rsm.*` — reciprocal-space mapping, VTK export           | xrayutilities, pyevtk                                       |
| `[notebook]` | self-contained Jupyter environment                       | ipywidgets, anywidget, ipyfilechooser, ipykernel, ipympl, jupyterlab, nbformat |
| `[all]`      | everything except dev                                    | `xdart[fitting,rsm,gui,notebook]`                       |
| `[dev]`      | test / build / release tooling                           | pytest, pytest-timeout, pytest-xdist, build, twine, tifffile, imagecodecs |

Extras compose. `[gui]` already bundles `[fitting]` + `[rsm]` — the GUI surfaces
Peak/Phase Fitting and the Grazing/GI/RSM workflow — so `pip install "xdart[gui]"`
(and the conda package) give you the **complete** GUI with no missing-dependency prompts.

> **Tip — use [`uv`](https://docs.astral.sh/uv/) if you have it.** It is a
> drop-in pip replacement that is typically 10–100× faster on cold installs.
> With the scientific-stack dependency tree (pyFAI, h5py, silx, PySide6, …)
> that is often the gap between a fresh-env install finishing in a few
> seconds vs. several minutes. `pip install uv` (or `brew install uv` /
> `winget install astral-sh.uv`), then prefix the commands with `uv `.

## Performance: install the HDF5 stack from conda-forge

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

## Output compression (lz4 default — reading `.nexus` outside xdart)

xdart writes the integrated 1D/2D stacks with **lz4+shuffle** by default (fast,
hdf5plugin filter 32004; ~gzip-class size). **Reading those `.nexus` files requires
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


## Editable install with Pixi (recommended)

On **Apple Silicon macOS or Linux x86-64**, use the repository's locked workspace:

```bash
git clone https://github.com/v-thampy/xdart.git
cd xdart
pixi install --locked
pixi run --locked xdart
```

`pyproject.toml` already declares `xdart = { path = ".", editable = true, ... }`
with the GUI, development, and notebook extras. Changes under `src/` are used
when you restart xdart; no separate `pip install -e` is needed. Always launch with
`pixi run --locked xdart` from this checkout to select it over a global release install.

To update an existing clone, save or commit your local work first, then run:

```bash
git pull --ff-only
git rev-parse --short=8 HEAD
pixi install --locked
pixi run --locked xdart
```

The Git hash identifies the exact source revision being tested. `--locked`
requires the lockfile to match the manifest and installs its recorded package
versions without updating the lockfile. It does not prevent editable source
changes. See the [Pixi run reference](https://pixi.sh/latest/reference/cli/pixi/run/).

The native stack comes from **conda-forge**: NumPy, h5py, hdf5plugin, HDF5,
Blosc, c-blosc2, lz4-c, Fabio, pyFAI, silx, PySide6, and xrayutilities. The local
xdart package is editable; remaining Python requirements are resolved by Pixi
through PyPI without replacing those conda packages. Inspect the environment
with `pixi list`, or confirm the source location with:

```bash
pixi run --locked python -c "import xdart; print(xdart.__file__)"
```

**Windows editable install:** the checked-in workspace currently locks only
`osx-arm64` and `linux-64`. Create a separate Windows workspace next to the clone,
so its environment and lockfile do not change the repository's platform lock:

```powershell
git clone https://github.com/v-thampy/xdart.git
mkdir xdart-dev
cd xdart-dev
```

Save this as `xdart-dev/pixi.toml` (the source checkout is the sibling `xdart`
directory):

```toml
[workspace]
name = "xdart-dev"
channels = ["conda-forge"]
platforms = ["win-64"]

[dependencies]
python = "3.13.*"
numpy = "*"
h5py = "*"
hdf5plugin = "*"
hdf5 = "*"
blosc = "*"
c-blosc2 = "*"
lz4-c = "*"
fabio = "*"
pyfai = ">=2026.5,<2026.6"
silx = "*"
pyside6 = "*"
xrayutilities = "*"

[pypi-dependencies]
xdart = { path = "../xdart", editable = true, extras = ["gui", "dev", "notebook"] }
```

Then run `pixi install` and `pixi run xdart` from `xdart-dev`. Keep the generated
`pixi.lock` to reproduce that Windows environment; subsequent launches can use
`pixi run --locked xdart`. To update the editable source, run `git pull --ff-only`
in the sibling `xdart` checkout. If its dependencies changed, run `pixi install`
in `xdart-dev` and retain the updated Windows lockfile. This Windows setup must
be validated on Windows before declaring a release supported there.

## Tests and packaging

From the repository workspace:

```bash
pixi run test                                    # headless core suite
pixi run gui-test                                # GUI suite, offscreen
pixi run python scripts/release.py check          # focused release preflight
pixi run python scripts/release.py build          # preflight, sdist/wheel, twine
```

For the separate Windows workspace, run Python/pytest with paths into the sibling
checkout. Release preflight is a focused check; it does not replace the CI suites
or native platform testing. Publishing is performed by the release workflows
when a maintainer pushes a version tag.

## Published releases and updates

A release becomes available after the maintainer pushes its version tag **and
the package publication workflows succeed**. A commit on `main`, or a local tag,
does not update the published package. Pixi global installs use the conda channel;
the installer scripts use PyPI. If an install was pinned to an older
version, its version constraint must also be updated.
