# xrd-tools Monorepo Plan

## Goal

Create one new repository and one published Python distribution:

```text
xrd-tools
```

with two import packages:

```python
import xrd_tools   # headless core, renamed from ssrl_xrd_tools
import xdart       # Qt GUI
```

Install shape:

```bash
pip install xrd-tools
pip install xrd-tools[gui]
```

The monorepo should start from stable, clean snapshots of the current
`refactor/architecture-v2` branches in `ssrl_xrd_tools` and `xdart`. Do not
migrate from dirty working trees.

## History: Import It (Recommended) or Start Fresh

Two options; importing history is recommended.

**Option A — import both histories (recommended).** The cost is modest
(~an hour) and is independent of the rename:

```bash
# In throwaway clones of each repo:
git filter-repo --to-subdirectory-filter src/xrd_tools    # ssrl_xrd_tools clone
git filter-repo --to-subdirectory-filter src/xdart        # xdart clone
# In the new xrd-tools repo:
git merge --allow-unrelated-histories <ssrl-clone>
git merge --allow-unrelated-histories <xdart-clone>
```

The `ssrl_xrd_tools -> xrd_tools` rename then happens as an ordinary commit on
top, so `git blame` and `git log --follow` keep working. This matters here more
than in most codebases: the code's review-trail comments (P2#4, BLOCKER-1,
T0-…, RS-…) get their context from history, and "the old repos remain the
record" means doing that archaeology across two frozen repos forever.

**Option B — start fresh.** A clean break is defensible because the migration
also changes the import name, the distribution model, the release process, and
the docs/tests layout. If chosen, the old repos remain the historical record
and `MIGRATION.md` records the exact source commits.

Either way, record the source SHAs in `MIGRATION.md`.

## Stage 0: Freeze Inputs

1. Finish testing `refactor/architecture-v2` in both repos.
2. Ensure both working trees are clean.
3. Record exact source commits:
   - `ssrl_xrd_tools/refactor/architecture-v2`
   - `xdart/refactor/architecture-v2`
4. Optionally tag or branch those commits as the migration source, e.g.
   `monorepo-source-2026-06`.
5. Do not include uncommitted work in the migration snapshot.

## Stage 1: Create Fresh Repo

Target layout:

```text
xrd-tools/
  src/
    xrd_tools/
    xdart/
  tests/
    core/
    xdart/
  docs/
    review/
  examples/
  scripts/
  licenses/
  pyproject.toml
  README.md
  MIGRATION.md
```

Copy:

```text
ssrl_xrd_tools/ssrl_xrd_tools  -> src/xrd_tools
xdart/xdart                    -> src/xdart
ssrl_xrd_tools/tests           -> tests/core
xdart/tests                    -> tests/xdart
review                         -> docs/review
example_notebooks              -> examples/notebooks
```

Copy the old licenses into `licenses/` and add a short root `LICENSE` noting
that the monorepo combines the former projects.

## Stage 2: Rename Core Imports

Mechanically replace:

```python
ssrl_xrd_tools -> xrd_tools
```

in:

- `src/xrd_tools`
- `src/xdart`
- tests
- examples
- docs where relevant

Keep:

```python
xdart
```

as the GUI import package.

Ship a minimal in-dist compatibility shim for one release cycle: a tiny
`ssrl_xrd_tools` package inside the same wheel whose `__init__.py` re-exports
`xrd_tools` and emits a `DeprecationWarning`:

```python
# src/ssrl_xrd_tools/__init__.py
import sys, warnings
import xrd_tools
warnings.warn("ssrl_xrd_tools is now xrd_tools; update imports.",
              DeprecationWarning, stacklevel=2)
sys.modules[__name__] = xrd_tools
```

This is nearly free, keeps every existing beamline notebook working through the
transition, and is much cheaper than the Stage 8 PyPI compatibility shells.
Remove it after one cycle. Fully adopt `xrd_tools` in all first-party code,
tests, and examples from day one — the shim is for *users'* notebooks only.

## Stage 3: Single Package Metadata

Use one root `pyproject.toml`.

Keep the base close to today's ssrl base. Do NOT promote the heavy
analysis extras into base: `pymatgen` alone drags sympy/spglib/monty/ruamel
and dominates install time and conda solves; `xrayutilities` is compiled.
The "analyses just work without discovering extra names" goal is met instead
by lazy imports with a friendly error ("pip install xrd-tools[fitting]") —
the pattern the codebase already uses — and by documenting
`pip install "xrd-tools[all]"` as the default for analysis workstations.

`pandas` belongs in base explicitly: `core/scan.py` and `reduction/core.py`
import it directly (it currently arrives only transitively via xarray).

```toml
dependencies = [
  "numpy",
  "scipy",
  "pandas",
  "xarray",
  "h5py",
  "nexusformat",
  "fabio",
  "silx",            # io.spec / io.metadata SPEC parsing
  "pyFAI>=2025.3,<2025.12",
  "joblib",
  "natsort",
  "matplotlib",      # viz.mpl (base today; consider lazy viz import later)
  "plotly",          # viz.plotly (same)
]

[project.optional-dependencies]
fitting = ["lmfit", "pymatgen"]
rsm = ["xrayutilities", "pyevtk"]
gui = [
  "PySide6>=6.5",
  "pyqtgraph>=0.13.7",
  "hdf5plugin",
  "qtawesome",
  "imagecodecs",
  "imageio",
  "pyyaml",
]
notebook = ["ipywidgets", "anywidget", "ipyfilechooser", "ipykernel", "jupyterlab"]
all = ["xrd-tools[fitting,rsm,gui,notebook]"]
```

(`ipykernel` moves out of base/GUI runtime deps — it was a questionable
runtime dep of xdart already.)

Script:

```toml
[project.scripts]
xdart = "xdart.xdart_main:main"
```

The `xdart` entry point installs even without `[gui]`, so `main()` must catch
the PySide6/pyqtgraph `ImportError` and print
`xdart requires the GUI extra: pip install "xrd-tools[gui]"` instead of a raw
traceback. This replaces the friendliness of the deleted runtime version guard.

Both import packages should report the same version from:

```python
importlib.metadata.version("xrd-tools")
```

## Stage 4: Remove Cross-Repo Machinery

Delete or rewrite:

- xdart's `ssrl_xrd_tools>=...` dependency floor;
- runtime minimum-ssrl-version guard;
- cross-repo publishing instructions;
- tests that compare xdart's dependency pin to ssrl's version.

Replace with:

- a core capability import test;
- architecture guard that `xrd_tools` never imports `xdart`;
- architecture guard that `xrd_tools.core`, `xrd_tools.io`,
  `xrd_tools.reduction`, and `xrd_tools.sources` do not import Qt.

## Stage 5: Tests and CI

Expect a small path-fixing pass where xdart tests reach for sibling-repo
fixtures (`../ssrl_xrd_tools`) and where `tests/core/conftest.py` assumed the
old repo root.

First focused pass:

```bash
python -m pytest tests/core/test_architecture_guards.py
python -m pytest tests/core/test_reduction.py tests/core/test_reduction_streaming.py
python -m pytest tests/xdart/test_core_capabilities.py
```

Broader pass:

```bash
python -m pytest tests/core
QT_QPA_PLATFORM=offscreen python -m pytest tests/xdart
```

Final gates:

- `import xrd_tools` works and does not import Qt.
- `import xdart` works.
- `xdart` CLI starts far enough to import and build the app shell.
- GI live == batch == reload spine still passes.
- Processed NeXus reload tests still pass.
- Image Viewer, XYE Viewer, and NeXus Viewer smoke tests still pass.

### CI (required before the first tag)

CI is the single biggest payoff of the monorepo — without it, the move
reproduces the no-CI status quo with a nicer layout. One GitHub Actions
workflow, running on every PR:

- `pytest tests/core -m "not slow"`;
- `QT_QPA_PLATFORM=offscreen pytest tests/xdart` (or at minimum
  `-m display_logic` plus the sink/adapter/writer-roundtrip subset if the full
  offscreen suite is too slow);
- the architecture guard tests explicitly (cheap, and their failure message is
  the one you most want on a PR);
- `pytest-timeout` enabled globally — two of the new streaming tests hang
  rather than fail when their fix regresses.

A `slow`/nightly job can run the full suites including the GI equivalence
spine. Add a release workflow later that builds, `twine check`s, and publishes
on tag.

## Stage 6: Documentation

Create `MIGRATION.md`:

```text
This repo was created from:
- ssrl_xrd_tools: refactor/architecture-v2 @ <sha>
- xdart: refactor/architecture-v2 @ <sha>
- review docs snapshot @ <date>

Package rename:
- ssrl_xrd_tools -> xrd_tools
- xdart unchanged
- distribution name: xrd-tools
```

Update README examples:

```python
from xrd_tools.reduction import run_reduction
from xrd_tools.sources import open_source
```

Update notebooks gradually unless tests depend on them.

## Stage 7: Release Model

One build and one release:

```bash
python -m build
twine check dist/*
```

One package:

```bash
pip install xrd-tools
pip install xrd-tools[gui]
```

One tag:

```text
v1.0.0
```

Use `v1.0.0`, not `v0.1.0`. This is a mature codebase — ~1300 tests, eight
releases past 0.35 — and `0.1.0` invites "is this alpha?" from exactly the
beamline users who should trust it. The rename is the natural moment to
declare 1.0.

PyPI name: `xrd-tools` is unclaimed (verified 2026-06-10 via
`pip index versions xrd-tools`). Register it early — upload a first build as
soon as the repo skeleton exists rather than waiting for the full migration.

### Distribution channel: pip + uv only (DECIDED 2026-06)

Conda-forge is dropped. The `ssrl_xrd_tools` feedstock PR (#32904) is closed
and the local `xdart` recipe is abandoned. Rationale: wheel coverage for the
full stack is now essentially complete (pyFAI incl. 2026.x ships Windows /
manylinux / Apple Silicon wheels; PySide6 vendors Qt; h5py vendors libhdf5;
hdf5plugin vendors the compression filters), conda-forge review latency is
days-to-weeks on volunteer capacity, and feedstock + rerender maintenance is
real ongoing cost for a solo maintainer. Nothing is foreclosed — a feedstock
can be added later at any time if demand appears. Conda *users* are
unaffected: `pip install xrd-tools` works inside a conda env.

Consequences for this plan:

- **`requires-python = ">=3.11"`** in the root pyproject — pyFAI ships no
  cp310 wheels for ≥2025.3, so allowing 3.10 means source builds.
- **Documented user install is uv:**

  ```bash
  uv tool install "xrd-tools[gui]"   # puts `xdart` on PATH, isolated env
  ```

  with `pip install "xrd-tools[gui]"` as the plain-pip equivalent. The
  Windows PS1 installer becomes a thin wrapper that bootstraps uv and runs
  the line above; `uv python install` removes any dependency on the user's
  system Python.
- **Known wheel gap:** `xrayutilities` has no Apple Silicon wheels (win_amd64
  and manylinux only). It is confined to the `[rsm]` extra, so mainstream
  installs are unaffected; macOS RSM users need Xcode CLT for the source
  build — add a one-line note in the README/install docs.

## Stage 8: Archive Old Repos

After the monorepo is stable:

- keep old repos read-only or clearly marked legacy;
- add README pointers saying development moved to `xrd-tools`;
- optionally publish compatibility shells later if users need them:
  - `ssrl_xrd_tools` depends on `xrd-tools`;
  - `xdart` depends on `xrd-tools[gui]`.

Do not make compatibility shells part of the first monorepo pass unless there is
a clear user need.

## Recommendation

Wait until `refactor/architecture-v2` is stable and committed in both repos,
then create the monorepo fresh from clean snapshots.

Use:

- one repository (with both histories imported via filter-repo);
- one distribution (`xrd-tools`, pip + uv only — no conda-forge);
- one version (starting at 1.0.0);
- one root `pyproject.toml` (slim base; `fitting`/`rsm`/`gui`/`notebook` extras);
- one test matrix, run by CI from day one;
- one release script.

Keep the code physically separated into `xrd_tools` and `xdart`, and enforce the
boundary mechanically with tests. Ship the 3-line `ssrl_xrd_tools` import shim
for one cycle so existing notebooks survive the rename.
