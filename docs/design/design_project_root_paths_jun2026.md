# Design — N1: Project Folder + portable relative paths (+ source-kind & embed-raw seams)

**Status:** AGREED design (Vivek signed off on the 4 N1 decisions + the source-seam framing, Jun 2026).
Implement on its **own branch** after the streaming/`data_2d` work on `refactor/architecture-v2` settles
(N1 is cross-cutting: wrangler params + disclosure + `nexus_writer` + ssrl readers). Two layers:
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
  - Enter Project Folder → the **PONI File** field appears (as today).
  - Enter a valid PONI → the **rest** of the tree appears (as today).
  - **Decision 2 — folder change RESETS, not just hides:** changing the Project Folder **clears** the
    dependent paths (PONI + everything downstream) and returns to the "enter PONI" state — because a PONI
    path relative to the *old* folder is meaningless under the new one. Not a cosmetic hide; an actual
    invalidation.
- **Decision 1 — out-of-tree paths:** a browsed PONI/raw/mask **inside** the Project Folder is stored
  **relative**; one **outside** it is stored **absolute** (with a `logger.warning`) — don't forbid
  browsing outside. Resolution handles both (relative → join to root; absolute → use as-is).
- **Decision 3 — missing root on session restore:** if a restored session's Project Folder no longer
  exists (moved machine), fall back to the **fresh-start disclosure** (prompt for the folder) instead of
  erroring; once re-entered, the relative paths resolve under the new root.

## 3. Data layer — relative storage + resolution (DECIDED 4: ships WITH the GUI)
The GUI field is only half of N1; the portability payoff is the file + reader. Both ship together.
- **Write:** persist the Project Folder as `entry/@source_base` (POSIX). Each frame's `source/path` is
  the **full** `PurePosixPath(os.path.relpath(src, root)).as_posix()` (depth-robust; POSIX separators for
  cross-OS). Absolute fallback (warn) when the raw is outside the root. Drop the current
  `os.path.abspath` at `nexus_writer.py:1053`.
- **Read:** resolve relative `source/path` against `@source_base` in the ssrl readers
  (`io/frame_view.py:_source_for_frame`, `core/scan.py:load_image`); add a `source_root=`/`data_root=`
  **override** on `read_scan`/`read_frame_view`/`open_scan`/`get_*` so a user who moved the data can
  repoint it. Precedence: explicit override > stored `@source_base` > the `.nxs` file's own directory.
  POSIX→native via `Path(PurePosixPath(stored))`.
- **Back-compat:** old `.nxs` with absolute `source/path` (no `@source_base`) keep loading (read by the
  absolute branch); old sessions with absolute paths load (treat as no-root until a folder is set).

## 4. Source-kind selector (DESIGNED-FOR, build with Tiled)
Above the Project Folder, a **source-kind selector** — but model it as a generic **source chooser** over
the existing `ssrl.sources` / `open_source(spec)` / `FrameSource` seam, **not** a hardcoded 2-way toggle
(so filesystem / tiled / future Bluesky sources all slot in). For now it has two entries:
- **Project Folder** (filesystem) — §2/§3 above; the spec is `{kind: filesystem, root: <folder>}`.
- **Tiled** — a **Tiled address/URI** field (remote-capable, not just localhost); the spec is
  `{kind: tiled, uri: <addr>, ...}`, routed to the (currently stub) tiled `FrameSource`.
The wrangler consumes a **source spec → `open_source(spec) -> FrameSource`**, never a bare filesystem
path — which is the keep-xdart-thin / WS-B direction. **Do not build the Tiled path now**; just ensure the
wrangler/plan path takes a source spec so Tiled is additive.

## 5. Embed-raw flag (DESIGNED-FOR, build with Tiled)
For a non-persistent source (Tiled), the raw path won't resolve off-site, so the `.nxs` must **embed the
raw**. Add a writer/sink flag **"store full raw" vs "store source ref"**:
- **Default off; auto-on for non-persistent sources** (Tiled). Optionally expose it for filesystem too as
  a general **"self-contained `.nxs`"** option (archiving).
- It's a **big** size hit (e.g. 651×18 MB ≈ 12 GB) → opt-in and **compressed** (writer already uses lzf).
- **Generalize the existing raw-storage path** (`skip_map_raw=False` already stores `map_raw` per frame)
  rather than inventing a new schema; the reader's `load_image` reads embedded raw when present, else the
  source ref. This is a **sink option**, so a headless notebook gets it too.

## 6. Sequencing & scope
- **N1 now (own branch, after streaming/`data_2d` settle):** §2 GUI Project-Folder + disclosure + §3
  data-layer (write relative + `@source_base`, read resolution + override, back-compat). Full round-trip
  + moved-tree + out-of-tree + missing-root tests.
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
