# Design — Plan B item 3: headless source/capability/provenance contracts migration

**Status (2026-07-01):** Stages 1–5 (the **headless half**) = **DONE + merged on `feature/remediation`**.
Stages 6–8 (the contested-GUI Plan B items 1/2/4) are **deferred** post-Phase-5 / a stable
`static_scan_widget.py` checkpoint. See the per-stage **Status checklist** at the bottom for commits.

**Convention:** every agent that lands a chunk of this plan updates the Status checklist at the
bottom of *this* doc after its chunk (stage tag → DONE + commit sha + one-line note). Keep the
checklist the single source of truth for where the migration stands.

---

## Purpose

Extract the **source / capability / provenance / readiness** decision contracts out of the xdart
GUI and into headless `xrd_tools`. This is the direct expression of the north star —
**headless-first `xrd_tools` APIs** (usable from notebooks/scripts/services with no Qt) and a
**thin xdart** shell over them (§ North star, `roadmap_2026-06-10.md`). Today these contracts are
Qt-free *logic* that happens to live under `src/xdart/…` (e.g. `controls_logic.py`,
`scan_plot_dialog.py`'s probe helpers, three copies of the incident-angle/monitor resolvers); a
headless caller cannot reach them without importing the GUI package.

This work is the **foundation** for two things:

1. The remaining Plan B GUI items (source-card readiness wiring, capability-driven run-gate,
   provenance disclosure) — they should consume *one* headless decision core, not re-implement it.
2. **Phase 5 / D3 (the one-store collapse)** — the eventual `FrameRecordStore`-first read path and
   the retirement of the `data_1d`/`data_2d` transitional mirrors. Getting the source/capability/
   provenance contracts headless first keeps the contested-GUI Phase-5 surface small and disjoint
   (see **Phase-5 sequencing** below).

**Scope guard.** Stages 1–5 are **behavior-preserving extractions** (MOVE + re-export shim, or a
guarded refactor). They are NOT the store collapse and do NOT touch the contested Phase-5 files.
The one behavior-*reconciling* step is Stage 5 (dedup the three resolver copies), and it is
explicitly sequenced after Phase-5 A-Step-A (see that stage).

---

## The 5 headless contracts

### (1) Readiness / capabilities / run-gate DECISION core — MOVE
The pure decision core that answers "is this source ready to run?", "what can this processed file
do?", and "why is the run button disabled?" currently lives in
`src/xdart/gui/tabs/static_scan/controls_logic.py` (Qt-free already; ~1972 lines). **Move** the
readiness/caps/run-gate decision functions (incl. `run_target_readiness_note`,
`controls_logic.py:1401`) into `xrd_tools/session/readiness.py`, leaving a **re-export shim** in
`controls_logic.py` so every existing import keeps working (notably Phase-5 A-Step-B's import of
`run_target_readiness_note` — see Phase-5 sequencing).

- **The fix that makes this a clean MOVE (Stage 1, below):** `controls_logic.py:873` currently does
  a lazy `from xdart.modules.reduction import _mask_for_plan` ("lazy, Qt-free legacy parity").
  Moving `controls_logic` under `xrd_tools` *as-is* would make `xrd_tools` import `xdart` — a north
  star violation (verified: `xrd_tools` imports **no** `xdart` today). So **also relocate the pure**
  `_mask_for_plan` + `_flat_mask_as_bool` (`src/xdart/modules/reduction.py:1081-1140`) into
  `xrd_tools` and re-point the moved readiness core at the `xrd_tools` copy. Add a **purity-guard
  test** asserting `xrd_tools` (and specifically `xrd_tools.session`) imports nothing under `xdart`.

### (2) Source probe → `xrd_tools/sources/probe.py`
`probe_first_frame(source)` and `raw_is_reachable(source)` today live in the GUI dialog module
`src/xdart/gui/tabs/static_scan/scan_plot_dialog.py:28,55`. They are pure `FrameSource` operations
(open frame 0, decide raw-reachability). **Move** them to `xrd_tools/sources/probe.py` and re-export
from the GUI call sites (`scan_source_widget`, `scan_plot_dialog`).

### (3) `describe_source_readiness` + `capabilities_for_processed` — composition
Compose (1) + (2) into two headless entry points:
- `describe_source_readiness(source, …)` — the readiness note for an *un-run* source.
- `capabilities_for_processed(metadata)` — the capabilities of an *already-processed* file, computed
  by **consuming `metadata['capabilities']`** (the disclosure the writer already stamps), **NOT** by
  re-opening HDF5 with an h5py `detect_capabilities` probe. The headless layer must not re-open files
  to guess what the writer already declared.
- **Preserve the true-live escape hatch.** A failed frame-0 probe must NOT flip `raw_reachable` to
  false for **live / unknown-length** sources (a true-live scan legitimately has no frame 0 yet at
  gate time). The composition keeps the existing "unknown length ⇒ don't demote" branch.

### (4) `build_reduction_config` + `NexusSink` emits `/entry/reduction/`
Close the **headless-vs-GUI provenance gap**: the GUI writer stamps `/entry/reduction/` provenance,
but a purely headless run does not get the same disclosure. Extract a `build_reduction_config(plan,
…)` in core and have the headless `NexusSink`/writer emit `/entry/reduction/` the same way the GUI
`nexus_writer` does (via `xrd_tools/core/provenance.py`, which today exposes `write_provenance`).
- **ADDITIVE + spine-gated.** This adds a group; it must not change existing bytes for the GUI path.
  Gate on the live≡batch≡reload equivalence spine — a headless-written file must round-trip identical
  to the GUI-written one for the shared datasets, with `/entry/reduction/` now present in both.

### (5) `resolve_incident_angle` + `resolve_monitor_norm` → `core/metadata.py` (dedup 3 copies)
The GI incident-angle resolution and the monitor-normalization resolution logic exist in **three**
copies (the guarded `_frame_norm` at `src/xdart/modules/reduction.py:1003`, plus the
incident-angle assembly around `reduction.py:265-374`, plus the plan-build paths). Extract plain
resolvers `resolve_incident_angle(...)` and `resolve_monitor_norm(...)` into
`xrd_tools/core/metadata.py` (which already exists) and have the copies delegate.
- This is a **behavior-RECONCILING** refactor (the copies differ subtly). **Pick the guarded
  `_frame_norm` form as canonical** (the finite/non-zero-guarded monitor path). Extract the plain
  resolvers **WITHOUT** a `StrictPolicy` argument — strictness is applied by the caller, not baked
  into the resolver. **Sequence the `reduction.py` edit off Phase-5 A-Step-A** (see below), so the
  reconciliation lands on top of a stable reduction-core checkpoint.

---

## The 8 stages

Tags: **MOVE** (relocate + shim) · **S/M** effort · **parallelizable** vs needs-deps ·
**contested-GUI** = touches a Phase-5 contested file (defer).

- **Stage 1 — MOVE `controls_logic` readiness/caps/run-gate core → `xrd_tools/session/readiness.py`
  + re-export shim.** *(contract 1.)* **[M / low risk, parallelizable, NOT contested-GUI]**
  Includes the fix: also relocate the pure `_mask_for_plan` + `_flat_mask_as_bool`
  (`reduction.py:1081-1140`) into `xrd_tools` so `controls_logic.py:873`'s lazy
  `from xdart.modules.reduction import _mask_for_plan` does **not** make `xrd_tools` import `xdart`.
  Add a **purity-guard test** (`xrd_tools.session` imports nothing under `xdart`). Keep the shim so
  `run_target_readiness_note` and friends still import from `controls_logic`.
- **Stage 2 — Extract `probe_first_frame` / `raw_is_reachable` → `xrd_tools/sources/probe.py`.**
  *(contract 2.)* **[S / low risk, parallelizable]** Re-export from the GUI dialog + widget.
- **Stage 3 — `describe_source_readiness` + `capabilities_for_processed` (composition).**
  *(contract 3.)* **[M / low-med risk]** **Needs Stages 1 + 2.** Consume `metadata['capabilities']`,
  not h5py `detect_capabilities`; preserve the true-live escape hatch.
- **Stage 4 — `build_reduction_config` + headless `NexusSink` emits `/entry/reduction/`.**
  *(contract 4.)* **[M / med risk, spine-gated]** ADDITIVE; gate on the equivalence spine.
- **Stage 5 — Dedup `resolve_incident_angle` + `resolve_monitor_norm` → `core/metadata.py`.**
  *(contract 5.)* **[M / med risk — behavior-RECONCILING]** Pick the guarded `_frame_norm` form as
  canonical; extract plain resolvers WITHOUT a `StrictPolicy` arg; **sequence its `reduction.py` edit
  off Phase-5 A-Step-A.**
- **Stage 6 — Plan B item 1** (source-card readiness wiring). **[contested-GUI — DEFERRED]** Land
  after a stable `static_scan_widget.py` checkpoint / after Phase-5.
- **Stage 7 — Plan B item 2** (capability-driven run-gate GUI). **[contested-GUI — DEFERRED]** Same.
- **Stage 8 — Plan B item 4** (provenance disclosure surfacing in the GUI). **[contested-GUI —
  DEFERRED]** Same.

Stages 6–8 touch the contested Phase-5 GUI files (`static_scan_widget.py`, `display_*.py`,
`image_wrangler_thread.py`, `qt_nexus_sink.py`) and are **deferred until a stable
`static_scan_widget.py` checkpoint / after Phase-5**, so they don't collide with the store collapse.

---

## Phase-5 sequencing

- **Stages 1–5 are DISJOINT from Phase 5's contested files** (`static_scan_widget.py`,
  `display_*.py`, `image_wrangler_thread.py`, `qt_nexus_sink.py`). They land **NOW** — they only
  touch `controls_logic.py` (add a shim), `scan_plot_dialog.py` (re-export), `reduction.py` (the
  pure-mask + resolver relocations), and new files under `xrd_tools/session|sources|core`.
- **The one coordination point** is Phase-5 **A-Step-B's import of `run_target_readiness_note`**:
  Stage 1's re-export shim keeps that import working, so Phase-5 A-Step-B does not need to move in
  lockstep. Land Stage 1 (with the shim) *before or independently of* A-Step-B; do not delete the
  shim until A-Step-B has been re-pointed at `xrd_tools.session.readiness`.
- **Stage 5's `reduction.py` edit is sequenced off Phase-5 A-Step-A** (the reduction-core
  checkpoint), so the resolver reconciliation lands on a stable base rather than racing A-Step-A.
- Stages **6–8** wait for a stable `static_scan_widget.py` checkpoint / after Phase-5 (contested-GUI).

---

## Status checklist (agents update after each chunk)

| Stage | Contract | Tag | Status | Commit | Note |
|-------|----------|-----|--------|--------|------|
| 1 | (1) readiness/caps/run-gate MOVE + mask relocation + purity guard | M / low / parallelizable / NOT contested | **DONE** | Stage 1 | `controls_logic` moved to `xrd_tools.session.readiness`; mask helpers moved to `xrd_tools.reduction.masks`; shims kept at `xdart.gui.tabs.static_scan.controls_logic` and `xdart.modules.reduction`, with `tests/core/test_readiness_purity.py`. |
| 2 | (2) probe extract | S / low / parallelizable | **DONE** | Stage 2 | Moved `probe_first_frame` / `raw_is_reachable` to `xrd_tools.sources.probe` with GUI shims. Added core behavior + import-purity coverage. |
| 3 | (3) composition (needs 1+2) | M / low-med | **DONE** | this commit | Added `xrd_tools.sources.readiness.describe_source_readiness` + `capabilities_for_processed`; consumes `metadata['capabilities']` and preserves the true-live failed-probe escape hatch. |
| 4 | (4) provenance builder + `/entry/reduction/` | M / med / spine-gated | **DONE** | this commit | headless NexusSink emits `/entry/reduction/`; GUI writer byte-identical via shared builder |
| 5 | (5) resolver dedup | M / med / RECONCILING | **DONE** | this commit | `resolve_incident_angle` / `resolve_monitor_norm` moved to `xrd_tools.core.metadata`; guarded `_frame_norm` monitor semantics are canonical; no `StrictPolicy` arg. |
| 5.5 | Display decision core MOVE | M / low / parallelizable / NOT contested | **DONE** | this commit | `display_logic` moved to `xrd_tools.session.display_logic`; xdart shim kept. The H7 read contract is now a headless API, so notebook/service callers can consume `resolve_frame_data` and `compute_display_state` without importing the GUI package. |
| 6 | Plan B item 1 (source-card readiness) | contested-GUI | **DEFERRED** | — | after stable `static_scan_widget.py` / Phase-5 |
| 7 | Plan B item 2 (run-gate GUI) | contested-GUI | **DEFERRED** | — | after stable `static_scan_widget.py` / Phase-5 |
| 8 | Plan B item 4 (provenance disclosure GUI) | contested-GUI | **DEFERRED** | — | after stable `static_scan_widget.py` / Phase-5 |
