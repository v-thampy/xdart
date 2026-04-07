# xdart
X-ray Data Analysis in Real Time

## Installation

A single installer script handles everything: it creates a dedicated conda environment, installs the heavy scientific stack from conda-forge (`pyFAI`, `h5py`, `pymatgen`, Qt, HDF5 libraries), and installs both `xdart` and its computational core [`ssrl_xrd_tools`](https://github.com/v-thampy/ssrl_xrd_tools) on top. This conda-for-native + pip-for-Python split avoids the binary mismatches that can occur when mixing sources for native-backed packages.

### One-line install (recommended)

No clone required — just run:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

This creates a new conda environment called `xrd` containing Python 3.12, the full scientific stack, `xdart`, and `ssrl_xrd_tools`. After it finishes:

```bash
conda activate xrd
xdart          # launch the GUI
```

> **Why `xrd`?** The script's default conda environment name is `xrd` — a short, memorable alias for "X-Ray Diffraction". This is the environment you'll activate whenever you want to use `xdart` or `ssrl_xrd_tools`. If you prefer a different name (e.g. to keep multiple versions side-by-side), pass `-n <name>`:
>
> ```bash
> curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- -n myenv
> ```

### If you don't have conda/mamba

Pass `--bootstrap` to have the script download and install [miniforge](https://github.com/conda-forge/miniforge) automatically into `~/miniforge3`:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- --bootstrap
```

### Installer options

```
-n, --name NAME       Conda environment name (default: xrd)
-p, --python VERSION  Python version (default: 3.12)
--bootstrap           Install miniforge to ~/miniforge3 if conda is missing
--branch BRANCH       Git branch to install from (default: dev)
--force               Replace an existing env of the same name
--no-xdart            Install ssrl_xrd_tools only, skip xdart
--dev                 Editable install (requires a local clone — see below)
```

### Updating

Re-run the installer with `--force`:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- --force
```

### Development setup

Developers who want an editable install with immediate reloads should clone the repos and run the installer locally with `--dev`:

```bash
git clone -b dev https://github.com/v-thampy/xdart.git
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
cd ssrl_xrd_tools
./scripts/install.sh --dev
```

The script will auto-detect a sibling `xdart` clone and install both in editable mode.

### Release branch

The installer currently points at the `dev` branch (both repos) while the APIs stabilize. Once the packages are more mature, the default will switch to `main`. To pin to a specific branch or tag at any time, pass `--branch <name>`.

### Conda-forge (coming eventually)

A conda-forge recipe is planned once the API stabilizes, which will make installation as simple as:

```bash
mamba create -n xrd -c conda-forge ssrl-xrd-tools xdart
```

Until then, the installer script above is the recommended path.

## Running

Once installed you can run the program by simply typing **xdart** in the terminal. **Important**: Make sure you have activated the conda environment xdart was installed in.

```bash
conda activate xrd
xdart
```
