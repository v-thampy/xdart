# Install + in-app update design (v1.0.x)

**Date:** 2026-07-08 · **Status:** decision doc + handoff spec. Companion artifacts:
`scripts/install_xdart.sh` / `scripts/install_xdart.ps1` (micromamba-based first drafts).

## 1. Requirements (maintainer)

- One pasted terminal line per platform installs everything for a NEW user with NO
  preinstalled conda/mamba/python.
- The compiled I/O stack must come from conda-forge (measured faster for Eiger
  bitshuffle/LZ4): `h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c`.
- xdart itself comes from PyPI (`xrd-tools[gui]`; conda-forge distribution was abandoned).
- A menu item inside xdart updates xdart in place.

## 2. Pixi findings (checked 2026-07-08 against current docs)

- Pixi WORKSPACES fully support mixed conda + PyPI dependencies in one manifest and ONE
  coherent solve (conda first, then PyPI via uv against the conda-resolved env). Exactly our
  layering, formalized, with a lockfile.
- **`pixi global` does NOT support PyPI packages** — the global-manifest schema has only conda
  `dependencies` (docs: "Dependencies are the Conda packages"). So `pixi global install xdart`
  is only possible if we publish a conda package of xrd-tools (own prefix.dev channel via
  rattler-build — nice later option: it would also give start-menu shortcuts via menuinst; NOT
  now).
- Therefore no tool collapses this to a no-script one-liner today. The one-line UX comes from a
  hosted install script either way.

## 3. Decision: keep the one-line script UX; use a pixi WORKSPACE inside it

The script (per platform) does:
1. Install pixi if absent (official one-liner installer, self-contained, no conda needed —
   answers the "users don't have mamba" concern: NOTHING is assumed; note the micromamba drafts
   also assumed nothing, they downloaded a static binary — pixi replaces that role with a
   better upgrade story).
2. Write `$APP_ROOT/pixi.toml` (APP_ROOT = `~/.local/share/xdart` / `%LOCALAPPDATA%\xdart`):

   ```toml
   [workspace]
   name = "xdart"
   channels = ["conda-forge"]
   platforms = ["osx-arm64"]        # script substitutes the host platform

   [dependencies]                    # conda-forge fast I/O stack
   python = "3.12.*"
   h5py = "*"
   hdf5plugin = "*"
   fabio = "*"
   hdf5 = "*"
   blosc = "*"
   c-blosc2 = "*"
   lz4-c = "*"

   [pypi-dependencies]
   xrd-tools = { version = "*", extras = ["gui"] }

   [tasks]
   xdart = "xdart"
   ```
3. `pixi install` in APP_ROOT (one solve, conda+PyPI consistent, lockfile written → support
   benefit: a user's `pixi.lock` reproduces their exact env when debugging).
4. Shim `xdart` onto PATH: `cd APP_ROOT && pixi run xdart "$@"` (sh) / `.cmd` equivalent
   (Windows) + write `$APP_ROOT/install_meta.json` (see §4).
5. README keeps the manual conda/mamba instructions as the "I already have conda" path; the
   script is the new-user path. Update the micromamba drafts to this shape (or keep micromamba
   as fallback if pixi's installer is unavailable on some beamline network — implementer's
   call; do NOT maintain both long-term).

Upgrade command (single, used by both the script re-run and the in-app updater):
`pixi update` inside APP_ROOT — refreshes the PyPI xrd-tools to latest AND keeps the conda
stack coherent in the same solve. (Pin `xrd-tools = ">=1,<2"` if v2 should not auto-arrive.)

## 4. In-app updater (Help → "Check for Updates…") — spec

**Metadata contract:** the installer writes `$APP_ROOT/install_meta.json`:
```json
{"flavor": "pixi-workspace", "app_root": "...", "update_cmd": ["<pixi>", "update"],
 "relaunch_cmd": ["<shim>/xdart"], "installed_by": "install_xdart.sh v1"}
```
xdart locates it by walking up from `sys.prefix` (the env lives under APP_ROOT) or via
`XDART_INSTALL_META`. No file found ⇒ pip/conda-manual/dev install ⇒ the menu item shows the
appropriate command in a COPYABLE dialog instead of executing (never auto-run pip in an
unknown env; detect editable installs via `direct_url.json`'s `dir_info.editable` and refuse
with a "development checkout — use git" message).

**Check:** menu action fires a small worker (QThread or the existing worker helper; NEVER the
GUI thread) fetching `https://pypi.org/pypi/xrd-tools/json` (timeout ~3 s), compares
`info.version` vs `importlib.metadata.version("xrd-tools")` with `packaging.version`. Offline ⇒
"could not check" toast, no error dialog. Optional: passive check at startup, badge-only.

**Update:** the dangerous part is upgrading a RUNNING env — on Windows loaded .pyd/.dll files
are locked (pip fails or half-writes); on macOS it works by inode semantics but a restart is
needed anyway. ONE cross-platform pattern: **update-on-exit**.
1. Dialog: "xdart 1.0.1 is available (you have 1.0.0). Update and restart?"
2. On confirm: `QProcess::startDetached` of a tiny updater helper (ship it as
   `xdart._updater`, runnable via the env's python) with args: parent PID, app_root,
   update_cmd, relaunch_cmd, log path (`$APP_ROOT/update.log`).
3. Helper: waits for parent PID to exit (poll, ≤60 s), runs update_cmd, writes the log,
   relaunches via relaunch_cmd on success; on failure leaves the log and relaunches the OLD
   version (pixi's transactional env update makes a half-updated env unlikely; pip-flavor
   fallback is best-effort).
4. The app, after launching the helper, does a normal clean close (the S-13-hardened path).
   Guard: refuse to start an update while a run is active (`_processing_active`) — reuse the
   run-lock predicate.

**Tests:** meta-discovery (found/absent/editable ⇒ execute vs copyable vs refuse); version
compare (newer/equal/pre-release); helper unit test with a fake parent PID + echo commands
(no network, no real pixi); the check-worker off-GUI-thread assertion. The PyPI fetch is
mocked; one manual live test before release.

**Size:** updater S/M (~half day incl. tests); script conversion to pixi S (~2 h).
**Sequencing:** post-tag (v1.0.1) — v1.0 ships with the script installers + README; the menu
item needs a PyPI release to update TO anyway.

## 5. README one-liners (final shape, post-publish; swap ORG)

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ORG/xrd-tools/main/scripts/install_xdart.sh | bash
```
```powershell
# Windows
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/ORG/xrd-tools/main/scripts/install_xdart.ps1 | iex"
```
Keep: "already using conda/mamba?" section with the existing manual instructions
(`conda install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c` +
`pip install "xrd-tools[gui]"`), and note the snapshot-publish strip list must KEEP
`scripts/install_xdart.*`.
