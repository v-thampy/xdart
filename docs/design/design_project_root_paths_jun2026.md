# Design — N1: Project Folder + portable relative paths (+ source-kind & embed-raw seams)

**Status:** SHIPPED in the v1.0 release (commit `f2e99a4a`, main). N1 layer (1) — the GUI Project-Folder
field + relative-path storage/resolution (§2 + §3) — is fully implemented and tested
(`tests/xdart/test_n1_writer.py`, `test_n1_disclosure.py`, `test_cli_session.py`;
`tests/core/test_n1_source_paths.py`). N1 layer (2) — the source-kind selector (§4) — has landed its seams
(`xrd_tools.sources.open_source` / `SourceSpec` / `SourceKind` / registry, consumed by the GUI
`scan_source_widget`); the embed-raw flag (§5) and a concrete Tiled `FrameSource` are still deferred (build
with Tiled). Two layers:
(1) the **GUI Project-Folder field + relative-path storage/resolution** — build now when N1 comes up;
(2) the **source-kind selector (Project Folder vs Tiled) + embed-raw flag** — *design the seams now,
build when Tiled is the active feature.* Keep xdart thin: all of this lives on the ssrl
`FrameSource`/`open_source` + sink seam; xdart only exposes the controls (so a notebook gets the same).

---

## 1. Goal
A processed `.nxs` is **portable** (resolves its raw images after the data moves machines) and a whole
xdart session is **relocatable** by setting one folder. The GUI surfaces a single **Project Folder** that
everything hangs off; the file stores **relative** raw paths against it. Forward-looking: support reading
from a **Tiled** server (no persistent file paths), which in turn needs an **embed-raw** option so the
`.nxs` is a self-contained unit.

## 2. GUI — Project Folder field + progressive disclosure (DECIDED)
- Add a **Project Folder** parameter (Qt folder dialog) at the **top of the wrangler parameter tree,
  above the `Calibration` branch.**
- **All other paths are relative to it:** PONI file, image file/dir, mask file, and the output
  (`h5_dir`), which **defaults to `[project_folder]/xdart_processed_data`**.
- **Progressive disclosure (extends the existing PONI-first flow):**
  - **Fresh start (no session restored):** everything in the tree is hidden **except Project Folder**.
    - **✓ Launch flags SHIPPED (v1.0, commit `f2e99a4a`).** `xdart -f` / `--fresh` enters this
      fresh-start state directly by skipping session restore (`load_session()` → `{}`,
      `save_session()` a no-op); `xdart -n NAME` / `--session NAME` restores a named session
      (`~/.xdart/NAME.json`). Both are wired in `_gui_main._apply_cli_session_args(sys.argv)`
      (`_gui_main.py:250`, called at the top of `run()` at `_gui_main.py:286`, before any widget loads)
      via the `XDART_SESSION_FRESH` / `XDART_SESSION_FILE` env gates (`utils/session.py:16-28`); the two
      flags are mutually exclusive. **NOTE (`f2e99a4a`):** `_apply_cli_session_args` sets
      `XDART_SESSION_FRESH` directly in `os.environ` (untracked by `monkeypatch`); tests must
      snapshot/restore it (a leak of this var caused the `test_controls_panel_v2` full-suite flake, now
      fixed by an autouse fixture in `tests/xdart/test_cli_session.py`). **Do not
      re-implement.** The Controls V2 fresh-start path now also leaves Save Path blank
      while Project Folder is unset, then defaults it under the selected project.
  - Enter Project Folder → the **PONI File** field appears (as today).
  - Enter a valid PONI → the **rest** of the tree appears (as today).
  - **Decision 2 — folder change RESETS, not just hides:** changing the Project Folder **clears** the
    dependent paths (PONI + everything downstream) and returns to the "enter PONI" state — because a PONI
    path relative to the *old* folder is meaningless under the new one. Not a cosmetic hide; an actual
    invalidation.
- **Decision 1 — out-of-tree paths:** a browsed PONI/raw/mask **inside** the Project Folder is stored
  **relative**; one **outside** it is stored **absolute** (with a `logger.warning`) — don't forbid
  browsing outside. Resolution handles both (relative → join to root; absolute → use as-is).
  **Implementation note (`f2e99a4a`):** the wrangler's `_compute_source_base` now REQUIRES the Project
  Folder to be an existing directory — `return path if os.path.isdir(path) else None`
  (`image_wrangler.py:1863`, `nexus_wrangler.py:430`). A blank OR non-existent folder yields
  `source_base=None` → the writer stores absolute paths (back-compat). The out-of-tree warning is
  de-duplicated per (source dir, root) so a whole scan does not re-log it
  (`read.py:483-490` `_OUTSIDE_ROOT_WARNED`).
- **Decision 3 — missing root on session restore:** if a restored session's Project Folder no longer
  exists (moved machine), fall back to the **fresh-start disclosure** (prompt for the folder) instead of
  erroring; once re-entered, the relative paths resolve under the new root.

## 3. Data layer — relative storage + resolution (DECIDED 4: ships WITH the GUI)
The GUI field is only half of N1; the portability payoff is the file + reader. Both ship together.
- **Write:** persist the Project Folder as `entry/@source_base` (POSIX). Each frame's `source/path` is
  the **full** `PurePosixPath(os.path.relpath(src, root)).as_posix()` (depth-robust; POSIX separators for
  cross-OS). Absolute fallback (warn) when the raw is outside the root. **[IMPLEMENTED]** The
  relative-vs-absolute logic now lives in `xrd_tools.io.read.relative_source_path(src, root)`
  (`read.py:463`), called from the core write path `write_frame_source_ref` / `write_frame_record`
  (`nexus_record.py:201-208`). The old `nexus_writer.py:1053` abspath no longer exists there (that line is
  now unrelated code); the writer stamps `@source_base` via `stamp_source_base` (`nexus_writer.py:1360`)
  and resolves the LiveFrame source through `_resolved_frame_source` (`nexus_writer.py`, near line 1413).
- **Read:** **[IMPLEMENTED]** relative `source/path` resolves against `@source_base` in the readers
  (`io/frame_view.py:_source_for_frame` at `frame_view.py:315`; `core/scan.py` `load_image`) via the shared
  `resolve_source_master` (`read.py:389`) / `_resolve_source_master` (`image_source.py:123`). The
  `source_root=` override is wired on `get_raw_frame` / `read_frame_view` / `iter_frame_views` /
  `load_processed_raw_or_thumbnail` and `open_source(nxs, source_root=...)` (`registry.py:72`). Precedence:
  explicit override > stored `@source_base` > the `.nxs` dir (`read.py:434-442`). POSIX→native via
  `Path(PurePosixPath(stored))`. (The doc's `data_root=` name was NOT chosen — the parameter is
  `source_root` everywhere.)
- **Back-compat:** old `.nxs` with absolute `source/path` (no `@source_base`) keep loading (read by the
  absolute branch); old sessions with absolute paths load (treat as no-root until a folder is set).

## 4. Source-kind selector (SEAM SHIPPED; Tiled entry still stub)
Above the Project Folder, a **source-kind selector** — modeled as a generic **source chooser** over
the `xrd_tools.sources` / `open_source(spec)` / `FrameSource` seam, **not** a hardcoded 2-way toggle.
**[IMPLEMENTED]** the seam has landed and evolved past the two-entry design:
`open_source(uri_or_spec, **opts)` (`registry.py:45`) dispatches a `SourceSpec(uri, SourceKind, options)`
(`core/scan.py`) across the registered kinds
`IMAGE_FILE / TIFF_SERIES / NEXUS_STACK / EIGER_MASTER / PROCESSED_NEXUS / SPEC / LIVE` (`registry.py:63-86`),
with `guess_source_kind` + `discover_scans` (`discover.py`) for auto-detect; the GUI `scan_source_widget.py`
consumes `open_source(spec)` (`scan_source_widget.py:439-443`) and forwards `source_root` for a moved tree
(`scan_source_widget.py:375`). `SourceKind.TILED` is declared (`core/scan.py:69`) but has NO registered
factory — `open_source` raises `ValueError` for it (`registry.py:86`); the concrete Tiled `FrameSource` is
still to build.

## 5. Embed-raw flag (DESIGNED-FOR, NOT built — still deferred to Tiled)
For a non-persistent source (Tiled), the raw path won't resolve off-site, so the `.nxs` must **embed the
raw**. Add a writer/sink flag **"store full raw" vs "store source ref"**:
- **Default off; auto-on for non-persistent sources** (Tiled). Optionally expose it for filesystem too as
  a general **"self-contained `.nxs`"** option (archiving).
- It's a **big** size hit (e.g. 651×18 MB ≈ 12 GB) → opt-in and **compressed** (writer uses the portable gzip+shuffle policy; an `lzf` request is aliased to gzip).
- **Generalize the existing raw-storage path** (`skip_map_raw=False` already stores `map_raw` per frame)
  rather than inventing a new schema; the reader's `load_image` reads embedded raw when present, else the
  source ref. This is a **sink option**, so a headless notebook gets it too.

## 6. Sequencing & scope
- **N1 (DONE, shipped v1.0 `f2e99a4a`):** §2 GUI Project-Folder + disclosure + §3 data-layer (write
  relative + `@source_base`, read resolution + `source_root` override, back-compat). Round-trip /
  moved-tree / out-of-tree / missing-root / old-file compat tests live in
  `tests/core/test_n1_source_paths.py`, `tests/xdart/test_n1_writer.py`, `test_n1_disclosure.py`,
  `test_cli_session.py`, `tests/core/test_session_api.py`.
- **Forward-looking (build with Tiled):** §4 source-kind selector + §5 embed-raw. Land the **seams** as
  part of N1 (wrangler consumes a source spec; writer has an embed-raw flag wired) so they're not a
  rewrite later; implement the Tiled source + auto-embed when Tiled becomes the feature.

## 7. Test plan (N1 gate)
- Round-trip same machine: write relative + `@source_base` → reload → `load_image` returns identical
  pixels to the pre-change absolute behavior.
- Moved tree: write under root A, move to root B, read with `source_root=B` → loads; without override →
  clear error naming the relative path + base.
- Out-of-tree: raw/PONI outside the folder → stored absolute, warned, still loads.
- Old-file/old-session compat: absolute-path `.nxs`/session still load + reintegrate.
- GUI: fresh-start hides all but Project Folder; PONI appears after folder, rest after PONI; **changing
  the folder resets** PONI + downstream; missing restored root → re-prompt.
- Equivalence spine + GI matrix green.

## 8. Open (none blocking) / notes
- The `@source_base` is the same concept as the GUI Project Folder — one root drives both the session and
  the file.
- PONI/mask provenance can stay absolute-or-relative as convenient; the **raw frame paths** are the
  portability pain and the required part. (Embedding raw, §5, sidesteps it entirely for Tiled.)
- Windows different-drive raw → absolute fallback (accepted).
