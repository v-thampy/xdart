# CC Handoff — Build the xrd-tools Monorepo (one session)

You are creating the new `xrd-tools` monorepo from the two existing repos and, while everything is
open, landing a set of bold foundation refactors. Source of truth for the migration mechanics:
`monorepo_plan.md` (this folder). Architectural rationale: `greenfield_design_2026-06-09.md`.
Issue references (S3, C2, C3, S4, S5, C5, C6, #78) are from `deep_review_2026-06-09.md`.

Work stage by stage. **Commit at the end of every stage with the full suite green.** A partial
session must still leave a working repo. If a Stage 6 item turns out deeper than expected, skip it,
note it in `MIGRATION.md` under "Deferred", and move on — Stages 0–5 are mandatory, Stage 6 items
are independent of each other.

## Guardrails (do not violate)

- **Do NOT `git push` or publish** — leave that to the maintainer.
- **Keep the NeXus writer/reader validators strict** (`validate_integrated_stack_write`,
  `_require_uniform_axes_1d/2d`, `_select_frames_to_write`). Never relax one to make a test pass.
- The **GI live≡batch≡reload equivalence spine** (`tests/xdart/test_gi_batch_real_data.py`) is the
  acceptance gate for anything touching the write/read path. A failing equivalence is a real bug.
- NeXus schema changes must remain **additive + back-compatible**; old files must still read.
- Everything outside Stage 6 is **behavior-preserving**. Stage 6 lists its allowed behavior changes
  explicitly; anything else you discover, report rather than change.
- When fixing a bug, label it **regression** (caused by this migration) vs **pre-existing**.

## Stage 0 — Preconditions (verify, don't assume)

1. Both repos: `refactor/architecture-v2` merged to `dev`, released (ssrl 0.41.0, xdart 0.40.0),
   working trees clean. Abort and report if not.
2. Confirm the two pre-release fixes from `fix_review_2026-06-10.md` are in the snapshot:
   (a) the streaming dispatch loop wraps `session.submit()` failures (no bare raise escaping
   `imageThread.run()`); (b) `gi_freeze_diagnostic` persists on the real GUI batch path. If either
   is missing, fix it in the old repo FIRST (it ships with the release), then snapshot.
3. Record both source SHAs; tag both as `monorepo-source-2026-06`.

## Stage 1 — Repo skeleton with imported history

Create `~/repos/xrd-tools`. Import BOTH histories (Option A in the plan):

```bash
# throwaway clones:
git clone ~/repos/ssrl_xrd_tools /tmp/mig-core && cd /tmp/mig-core && git checkout dev
git filter-repo --to-subdirectory-filter . --force   # then move: see below
```

Use `git filter-repo --to-subdirectory-filter` so the ssrl tree lands at repo root paths that
match the final layout as closely as possible (`ssrl_xrd_tools/ssrl_xrd_tools` → `src/xrd_tools`
happens in Stage 2 as a normal `git mv` commit so blame survives; the subdirectory filter just
needs to prevent path collisions between the two histories). Then in the new repo:
`git merge --allow-unrelated-histories` each clone. Verify `git log --follow` works on a couple of
deep files (e.g. the reduction core) before proceeding.

Target layout (from the plan):

```text
xrd-tools/
  src/xrd_tools/   src/xdart/   src/ssrl_xrd_tools/  (3-line shim)
  tests/core/      tests/xdart/
  docs/review/     examples/notebooks/
  scripts/         licenses/
  pyproject.toml   README.md   MIGRATION.md
```

`docs/review/` = snapshot of `~/repos/review`. Copy old LICENSEs into `licenses/` + a short root
`LICENSE` noting the combination.

## Stage 2 — Rename + shim

1. `git mv` into the final layout (one commit, no content changes).
2. Mechanical rename `ssrl_xrd_tools` → `xrd_tools` across `src/`, `tests/`, `examples/`, docs.
   One commit. Keep `xdart` as the GUI import name.
3. Add the user-facing shim `src/ssrl_xrd_tools/__init__.py` (re-export `xrd_tools` via
   `sys.modules`, `DeprecationWarning`). First-party code never imports it; add a guard test.

## Stage 3 — Single package metadata

One root `pyproject.toml` exactly per the plan's Stage 3 block: slim base (incl. `pandas`
explicitly), extras `fitting` / `rsm` / `gui` / `notebook` / `all`, **`requires-python = ">=3.11"`**,
`pyFAI>=2025.3,<2025.12`, version `1.0.0`, `[project.scripts] xdart = "xdart.xdart_main:main"`.
`main()` must catch the Qt-stack `ImportError` and print
`xdart requires the GUI extra: pip install "xrd-tools[gui]"`. Distribution is pip + uv only — no
conda anything. Verify any package data (icons, default yaml/config files) survives the build:
`python -m build && twine check dist/*`, then install the wheel in a scratch venv and run the
import gates below.

## Stage 4 — Delete cross-repo machinery; strengthen guards

Delete: the `ssrl_xrd_tools>=` floor, the runtime min-version guard in `xdart_main.py`, the
floor-sync test, PUBLISHING.md's two-package choreography. Replace with:

- architecture guard: `xrd_tools` never imports `xdart` (port the AST test);
- import guard: `xrd_tools.{core,io,reduction,sources}` pull in no Qt/pyqtgraph;
- a core capability-import test (`from xrd_tools.io import relative_source_path`, etc.);
- guard: nothing under `src/` imports the `ssrl_xrd_tools` shim.

## Stage 5 — Green + CI

1. Fix test-path fallout (sibling-repo fixture paths, conftest roots). Full suite green:
   `pytest tests/core` and `QT_QPA_PLATFORM=offscreen pytest tests/xdart`.
2. GitHub Actions: PR workflow = core suite (`-m "not slow"`), offscreen xdart suite (or the
   `display_logic` + sink/adapter/writer-roundtrip subset if runtime forces it), architecture
   guards as a named job, `pytest-timeout` global. Nightly = full suites incl. the GI equivalence
   spine. Release workflow = build + `twine check` on tag (no auto-publish).
3. Final gates: `import xrd_tools` (no Qt in `sys.modules`); `import xdart`; `xdart` CLI reaches
   the app shell offscreen; equivalence spine green; processed-NeXus reload green; viewer smoke
   tests green.

**Stages 0–5 are the mandatory core. Everything below is the bold part — do as much as fits,
each item its own commit, suite green between items.**

## Stage 6 — Foundation refactors (riskier, more rewarding — in this order)

**6a. Move the complete-v2-record orchestration into the core (S3 + C2 + C3 — the big one).**
Today `xdart/modules/ewald/nexus_writer.py` owns per-frame source refs
(`frames/frame_NNNN/source/{path,frame_index}`), the `@source_base` stamp + conflict check,
thumbnails (`_quantize_thumbnail`), per-frame geometry upserts, provenance stamping, and
`_drop_integrated_rows`. Move all of it into `xrd_tools.io.nexus` as public primitives
(`write_frame_source_ref`, `stamp_source_base`, `write_thumbnail`, `drop_integrated_rows`, …) and
extend the headless `NexusSink` so a **purely headless run writes the complete v2 record** —
`get_raw_frame`, `read_frame_view` geometry, and N1 portability must work on a headless-written
file (add exactly that round-trip test in `tests/core`). xdart's writer becomes orchestration-free:
it calls the core primitives and keeps only the Qt/GUI-side concerns (append cursor, NFS retry,
signals). Rewrite the core N1 tests to use the real writer functions instead of hand-built h5py
fixtures. **Gate:** equivalence spine + full writer-roundtrip suites green; files written before/after
byte-compatible for an identical scan (assert on a fixture).
This is deliberately the first step toward the `xrd-session` data-ownership layer
(`greenfield_design` Difference 2) — name the new module section accordingly
(`xrd_tools/io/nexus_record.py` or similar), don't just paste functions.

**6b. Schema-as-code starter (Difference 5).**
Add a declarative `SCHEMA` structure in `xrd_tools.io` — schema version, the row-aligned dataset
set, capability attribute names — and make `drop_integrated_rows`, the writer validators, and the
reader version check consume it. Extend `warn_if_newer_schema` to `get_1d/get_2d/get_thumbnail/
get_frames` (the gap noted in `fix_review_2026-06-10.md`). Additive only; no on-disk change.

**6c. API renames while the rename is free** (allowed behavior changes, do all four):
- `io.read.Scan` → `ProcessedScan` (S5); keep `Scan = ProcessedScan` alias in `xrd_tools.io` with
  a deprecation comment, since the shim already breaks the old import path anyway.
- Delete the ~275-line dead legacy `Frame/MaskSpec/FrameSource/Scan` block in
  `reduction/core.py:136–411` (S4); keep the aliases to the core contracts.
- Energy sentinel: `None`, never NaN, across `io.read` (`energy_keV`/`energy_eV`/`get_metadata`)
  (#78). Convert at the read boundary; update the type hints to match reality.
- Remove the dead `NexusSink.swmr` field + its dead atomic-mode term (S6 residue).

**6d. Kill the duplicate seam code (C5 + C6).**
Either route `open_live_reduction_session` through `LiveScanFrameSource.to_scan()` (one adapter)
or delete `LiveScanFrameSource` and fix ARCHITECTURE_V2.md — pick whichever the tests make
cheaper, but end with ONE LiveScan→core adapter and one copy of `_scan_data_row`/path/wavelength
extraction. Re-point xdart's `two_d_kind_from_units` at `xrd_tools.core.frame_view`'s enum version
(relax the display_logic purity guard to allow `xrd_tools.core`, which is Qt-free — update the
purity test's allowlist, and map enum→legacy strings at the display edge).

**6e. If time remains:** S8 warning granularity (key monitor warnings per scan, not per process);
version-probe/`retain_products` cleanup notes from `fix_review_2026-06-10.md`; the 3 stray
`print()`s in `gui/widgets/`; stale "TEST-ONLY" comment block in `image_wrangler_thread.py`.

## Stage 7 — Close out

`MIGRATION.md`: source SHAs, rename table, dependency-layout changes, Stage 6 items completed vs
deferred, and any behavior changes shipped (the 6c list). README: new install
(`uv tool install "xrd-tools[gui]"`), new imports, badge for CI. Update `CLAUDE.md` (merge the two
old ones: one Environment section, the guardrails, the package map with `xrd_tools` paths, the
display-layer notes). Do NOT tag; leave `v1.0.0` tagging to the maintainer.

## Final acceptance checklist

- [ ] Both histories imported; `git log --follow src/xrd_tools/reduction/core.py` reaches pre-migration commits
- [ ] Full suite green: `tests/core` + offscreen `tests/xdart`
- [ ] `import xrd_tools` works in a wheel-installed scratch venv without Qt installed
- [ ] `pip install <wheel>` (no extras) + `xdart` prints the friendly gui-extra message
- [ ] Headless-written `.nxs` passes the complete-record round-trip (6a test)
- [ ] Equivalence spine green
- [ ] CI workflows present and the PR workflow passes locally via `act` or by inspection
- [ ] MIGRATION.md records everything deferred
