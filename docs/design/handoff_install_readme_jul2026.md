# Handoff — installer commits, README restructure, remaining pre-tag sweep

**Date:** 2026-07-08 · **For:** implementing agent (terminal access) · **From:** review orchestrator.
Context docs: `design_install_and_update_jul2026.md` (install strategy + in-app updater spec),
memory of the live verification below. Everything here is v1.0-tag-adjacent; nothing touches the
reduction spine.

## 0. What happened (context you need)

The install strategy moved to a **pixi workspace** driven by one-line scripts
(`scripts/install_xdart.sh` / `.ps1`), fully verified end-to-end on the maintainer's Mac against
the local checkout (`XDART_LOCAL_SOURCE` hook). Verified facts: pixi's installer respects
`PIXI_HOME` + `PIXI_NO_PATH_UPDATE` (fully self-contained, no shell-rc edits); one solve installs
the conda-forge fast I/O stack (h5py/hdf5plugin/fabio/hdf5/blosc/c-blosc2/lz4-c) + PyPI
`xrd-tools[gui]`; python 3.13 default (A/B vs 3.12: 117.4 vs 120.0 s on the 3621-frame Eiger
benchmark, both lz4 — tie, 3.13 kept for support horizon); `xdart` shim in `~/.local/bin`.
Three real failure modes were hit live and are now guarded IN the script: silent uncompressed
writer fallback (LZ4 round-trip check — root cause was a stale `XDART_INTEGRATED_COMPRESSION` env
var, NOT hdf5plugin), missing GUI stack (import check), and old-install PATH shadowing + shell
hash caching (`type -ap` detection + `hash -r` guidance, silent on clean machines).

## 1. Commit the pending work

`git status` and commit what's uncommitted, grouped (adjust if some already landed):

1. `scripts/install_xdart.sh` + `scripts/install_xdart.ps1` —
   `scripts: pixi-workspace one-line installers (py3.13 default, LZ4+GUI sanity checks, extras/local-source hooks, shadow+hash detection)`
2. `docs/design/design_install_and_update_jul2026.md` —
   `docs: install strategy (pixi workspace) + in-app updater spec (v1.0.1)`
3. `docs/design/design_gui_display_robustness_jul2026.md` (if still untracked) —
   `docs: GUI display robustness design (fragilities F-A..F-F, v1.1 plan V1-V6)`
4. `docs/design/handoff_install_readme_jul2026.md` (this file).
5. README changes from §2 below — own commit.

## 2. README restructure (the pixi switch, explained)

**Why the switch:** the old README told users to install conda/mamba, create an env, install the
fast conda stack, then pip-install the package — four manual steps with per-user drift. The pixi
workspace does the SAME layering (conda-forge compiled I/O stack + PyPI xrd-tools, which is why
it's fast) in ONE solve with a lockfile, bootstrapped by one pasted line, with `pixi update` as
the built-in upgrade path (and the future in-app updater's hook). Nothing about the package
changed — only how the environment is assembled. Conda/mamba users lose nothing; their path stays
documented and supported.

**New Install section order (top to bottom):**

### 2.1 Quick install (recommended) — PROMOTE TO TOP
```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ORG/xrd-tools/main/scripts/install_xdart.sh | bash
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/ORG/xrd-tools/main/scripts/install_xdart.ps1 | iex"
```
Copy: needs nothing preinstalled (no conda, no Python); installs into its own folder
(`~/.local/share/xdart` / `%LOCALAPPDATA%\xdart`) without touching existing Python/conda setups;
uses conda-forge builds of the HDF5/compression stack (fastest Eiger decode) + xrd-tools from
PyPI; launch with `xdart`; **upgrade by re-running the same line**. Optional extras:
`XDART_EXTRAS="gui,fitting"` before the command. Troubleshooting one-liner: if `xdart` launches
an old version, run `hash -r` or open a new terminal (the installer prints specifics when it
detects this). REPLACE `ORG` with the real org at publish; the URL works only once the repo is
public.

### 2.2 Using conda/mamba (existing setups) — KEEP, demote to second
Preserve the current instructions verbatim in spirit:
```bash
conda install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c
pip install "xrd-tools[gui]"
```
One-line note: this is the same layering the quick installer automates.

### 2.3 Plain pip — KEEP
`pip install "xrd-tools[gui]"` works everywhere; note Eiger bitshuffle/LZ4 decode is measurably
slower with PyPI-wheel HDF5 than the conda-forge builds, and lz4-compressed outputs need
hdf5plugin (a base dep) to read outside xrd-tools.

### 2.4 Headless / notebooks with pixi — NEW section (see §4 for content)

### 2.5 Development — KEEP
Clone + editable install; mention
`XDART_LOCAL_SOURCE=/path/to/checkout bash scripts/install_xdart.sh` as the way to test the full
user install against a working tree.

**Also update:** any remaining "install mamba/conda first" framing outside the Install section;
the snapshot-publish strip list must KEEP `scripts/install_xdart.*` and the new README text.

## 3. Remaining pre-tag sweep (verify each, don't assume)

- [ ] All fix-wave work committed; ledgers current (installer learnings row optional).
- [ ] Two one-token product improvements surfaced by the installer testing (v1.0 if trivial,
      else first v1.1): (a) `resolve_stack_compression`: EMPTY-string env value currently maps to
      None silently — empty should mean "unset → default" (only explicit `none`/`off` disables);
      (b) the `compression = None` announcement should be WARNING, not INFO, when it comes from
      the env var (silent 4x-bigger files was the observed failure).
- [ ] Re-freeze at final SHA → full battery (core + all offscreen suites + spine w/ real data +
      byte-compat) → record the table.
- [ ] Session-1 residue: sweep the ledger for `fixed-unverified` rows NOT incidentally
      live-verified during the perf/testing marathon; G18 real-data χ validation (incl. explicit
      partial-wedge + q-wedge cases) is the hard gate — it protects written bytes.
- [ ] RC-8: merge → snapshot-publish (strip list per `ae4292fb` + keep scripts/ + README) → tag
      v1.0.0 → build → `twine check` → `twine upload` → stub-and-archive old PyPI names
      (`xdart`, `ssrl-xrd-tools`).
- [ ] Post-publish smoke: run the REAL one-liner (registry path — first exercise of
      `xrd-tools = {version = ">=1,<2", extras=[...]}` through pixi/uv) on a clean account/VM;
      run the Windows `.ps1` once on a Windows box/VM (its JSON path-escaping in
      `install_meta.json` is untested — eyeball the emitted file).
- [ ] v1.0.1 queue: in-app updater per `design_install_and_update_jul2026.md` §4 (reads
      `install_meta.json`, update-on-exit helper); README badge/links.

## 4. Headless / notebook use with pixi (maintainer's question — YES)

A pixi workspace replaces the "mamba env with everything installed" per analysis project, same
fast stack, plus a lockfile that makes the analysis reproducible. Recipe (goes in the README §2.4
and/or the example-notebooks doc):

```bash
mkdir my-analysis && cd my-analysis
pixi init
pixi add python=3.13 h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c jupyterlab
pixi add --pypi "xrd-tools[notebook,fitting]"
pixi run jupyter lab
```

Notes for the section: the env lives in `./.pixi/` next to the notebooks; `pixi.toml` +
`pixi.lock` checked into the analysis folder = anyone (including future-you) reproduces the exact
env with one `pixi install`; add `[rsm]` via conda where possible
(`pixi add xrayutilities` from conda-forge — avoids the missing mac-arm PyPI wheels);
`pixi run python script.py` for headless batch scripts; existing conda/mamba envs keep working —
this is an option, not a migration.

**Shared beamline env + VS Code (maintainer's deployment pattern — include in the README):**
a pixi env is a normal prefix (`<workspace>/.pixi/envs/default/bin/python`), so the
one-shared-env / many-user-directories pattern maps 1:1. Recipe: ONE pixi workspace at a shared
path (e.g. `/shared/xrd-env/` = pixi.toml + pixi.lock + .pixi/); users open their own notebook
folders in VS Code and select the shared env's `bin/python` as interpreter/kernel (VS Code
auto-discovers pixi envs; manual "Enter interpreter path" always works). Register it by name in
every kernel picker once:
`cd /shared/xrd-env && pixi run python -m ipykernel install --prefix /usr/local --name xrd-tools --display-name "XRD Tools (shared)"`.
Admin updates with `pixi update` in that directory; the lockfile makes the shared env rebuildable
identically on a new machine (`pixi install`). Terminal equivalent of `conda activate` =
`pixi shell` in the workspace dir.

## 5. Execution notes

Work on `feature/remediation`. Gates: README/docs/scripts changes need no test battery, but run
`bash -n scripts/install_xdart.sh` (syntax) and, if feasible, one `XDART_LOCAL_SOURCE` install on
the dev machine after any script edit. Update this file's checkboxes as you go (repo convention).
