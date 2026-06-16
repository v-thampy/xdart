# xrd-tools

<!-- After the repo is pushed, point the badge at the real org/name:
[![PR checks](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml/badge.svg)](https://github.com/<org>/xrd-tools/actions/workflows/pr.yml) -->

SSRL X-ray diffraction toolkit: a **headless reduction core** (`xrd_tools`)
and the **xdart Qt GUI** (`xdart`) in one distribution.  Formerly the
`ssrl_xrd_tools` and `xdart` repositories — merged with full histories;
see [`MIGRATION.md`](MIGRATION.md).

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
