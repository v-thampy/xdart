# ADR-0006: `prepare_gi_freeze(source, plan)` — whole-scan incidence-extent discovery moves into core

**Status:** accepted · 2026-06-13 · (greenfield Difference 2 — the "thin xdart" gap; resolves codex-P1)
**Builds on:** ADR-0002 (capability attrs), ADR-0003 (does not touch event/record cardinality).
**Scopes against:** the project guardrail "do not over-abstract for non-existent features."

## Context

Difference 2 from all three deep reviews is the headline "thin xdart" gap: xdart still
owns headless reduction logic that the library should. The clearest instance is the
**grazing-incidence (GI) whole-scan grid freeze**. To produce a stack the writer's
uniform-axes validators accept, every GI frame in a batch must be integrated onto **one
common output grid** that brackets the scan's full incidence range — not per-frame axes
that drift and get rejected mid-run. The freeze *math* is already headless and already
correct: `_apply_gi_freeze_policy` (`src/xrd_tools/reduction/core.py:1576`) integrates a
small set of **scout frames** and unions their axes via `freeze_common_axis` /
`freeze_common_axes_2d` (lines 1632/1658), writing the result into
`plan.integration_1d/2d` (lines 1639/1662). Which frames it scouts is already a knob:
`_gi_freeze_scout_indices` (line 1691) reads `plan.extra["gi_freeze_scout_indices"]`
**first** (line 1697), validates membership fail-loud (line 1703-1704), and only falls
back to `[first, last]` (line 1708-1710) when absent.

**The blocker (codex P1).** `ReductionSession` builds its `Scan` from the *current chunk*.
In xdart's live-streaming batch path, that is **chunk 1**. So the session's default
`scout_union` brackets chunk 1's incidence range and **clips** any later, higher-incidence
frame — a silent data-quality regression on exactly the angle-dependent Eiger scans where
incidence varies most. The session cannot know the whole scan's extent because it never
saw the whole scan.

**xdart's current workaround** is `_gi_freeze_whole_scan_prepass` and friends in
`src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py` (~lines 1073-1399):
it walks the filesystem for the scan's files (`_enumerate_scan_files`, :1373), reads
per-frame metadata to find the **global** lowest- and highest-incidence frames
(`_resolve_incidence_from_meta` :1359; the decision tree in `_gi_whole_scan_scout_entries`
:1259), loads only those two images, and freezes over them via the boundary adapter
`freeze_live_scan_gi_ranges` (`src/xdart/modules/reduction.py:447`) — which opens a
throwaway freeze-only session with `chunk_size=len(frames)` (:488) and copies the frozen
ranges into `scan.bai_*_args` (`_copy_frozen_gi_ranges_to_live_scan`, :501). The streaming
session then opens with `gi_freeze_mode=None` and consumes the pre-frozen grid.

The genuinely-misplaced part is **whole-scan incidence-extent discovery**: it is pure,
Qt-free, image-free metadata logic that lives in a Qt thread only because that is where
the GUI's source attributes are. It is the one thing core cannot do today, because core
has no way to ask a source "what is your whole-scan metadata?" without loading images.

The three reviews and a codex pass also claimed this freeze is the *first instance* of a
general lifecycle stage — `source → prepare(source, plan) → per-frame → finalize` — that
RSM grid extent, stitch bounds, common-mask, and the Auto 1D/2D range buttons would all
share. **A code audit refutes that for today:** auto-range is GUI checkbox state
(`integrator.py`, no enumeration), common-mask does not exist (zero references in the
tree), RSM `scout()` is a caller-wired library call (`rsm/gridding.py`, no session
prepass), and stitch is a *finalize*-stage post-pass (`modules/ewald/stitch.py`, all
frames already in memory). Generalizing now would abstract over **one real user and three
non-users**, which the project guardrail forbids.

## Decision

**Move whole-scan incidence-extent *discovery* into core as a single concrete function,
`prepare_gi_freeze(source, plan)`, fed by a new optional `FrameSource` capability
`scan_manifest()`. Do NOT build a `prepare()` provider framework, a `FrozenPlan` wrapper,
or a lifecycle registry.** Concretely:

### 1. One additive capability flag — `core/scan.py:41`

```python
@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    ...
    has_scan_manifest: bool = False  # scan_manifest() yields whole-scan
                                     # per-frame metadata CHEAPLY (no image reads)
```

Default `False` ⇒ byte-identical for every existing source, including `Scan.capabilities`'
default (`core/scan.py:250`). This single flag **is** the warn-and-proceed lever: a source
that cannot cheaply enumerate leaves it `False`. We deliberately use **one** flag, not two:
"manifest exists but `< 2` readable incidences" is a *runtime scout outcome*, not a static
source property, so it belongs in the diagnostic, not a second capability.

### 2. `scan_manifest()` — optional protocol method, concrete default on `BaseFrameSource`

Add to the `FrameSource` Protocol (`core/scan.py:206`) and give `BaseFrameSource`
(`sources/base.py`, next to `metadata_for` at :55) a concrete default:

```python
def scan_manifest(self) -> list[tuple[int, Mapping[str, Any]]] | None:
    """Cheap METADATA-ONLY pass: (frame_index, metadata) for every frame.
    MUST NOT load detector images. Return None when whole-scan metadata
    cannot be cheaply enumerated (caller treats None as 'unverifiable')."""
    if not self.capabilities.has_scan_manifest:
        return None
    return [(idx, dict(self.metadata_for(idx))) for idx in self.frame_indices]
```

`None` (not `[]`) unambiguously means "do not even try; warn-and-proceed," distinct from a
real empty scan. `TiffSeriesSource` / `ImageFileSource` flip `has_scan_manifest=True` only
when `metadata_format is not None` — the **same** condition that already gates
`has_metadata` (`sources/image.py:49`, :105). `NexusStackSource`, Eiger masters, and
Image-Directory keep `False` ⇒ `scan_manifest()` returns `None` ⇒ warn-and-proceed,
byte-identical to today's `_enumerate_scan_files() == []` branch.

**`meta_dir` must be threaded** (verification fix): `read_image_metadata(path, fmt,
meta_dir)` uses `meta_dir` as load-bearing for SPEC sidecars stored separately from images.
`TiffSeriesSource.metadata_for` (`sources/image.py:133`) currently drops it. `scan_manifest()`
must read metadata from the same location xdart's scout does, or a separate-SPEC source
yields different/empty incidences ⇒ different extremes ⇒ broken equivalence. Add an
optional `meta_dir` to `TiffSeriesSource`/`ImageFileSource` and pass it through.

### 3. `prepare_gi_freeze()` + `PrepareDiagnostics` — `reduction/core.py`, near `_apply_gi_freeze_policy`

`PrepareDiagnostics` is the **only** new type:

```python
@dataclass(frozen=True, slots=True)
class PrepareDiagnostics:
    status: str                            # "frozen" | "skip" | "unverifiable"
    reason: str = ""
    scout_indices: tuple[int, ...] = ()
    scout_refs: tuple[Mapping[str, Any], ...] = ()  # resolved source refs for the
                                                    # extremes (path/file-number/meta),
                                                    # so the caller's loader is a plain
                                                    # read_image — not a re-enumeration

def prepare_gi_freeze(
    source, plan, *, freeze_policy="scout_union", incidence_motor=None,
) -> tuple[ReductionPlan, PrepareDiagnostics]:
    """WHOLE-SCAN prepass. Scout the source's full metadata extent and return a COPY of
    plan with extra["gi_freeze_scout_indices"] pinned to the GLOBAL incidence extremes
    (or left unchanged). The returned plan is NOT yet frozen — hand it to the freeze step
    (ReductionSession w/ scout_union, or xdart's freeze_live_scan_gi_ranges) and the
    EXISTING machinery freezes over the pinned indices instead of chunk-1's first/last.
    GI-only: non-GI plans pass through with status='skip'. Never raises for an
    unenumerable source (returns status='unverifiable')."""
```

**Critical design call — `prepare_gi_freeze` computes WHICH FRAMES, never integrates and
never loads images.** It uses three private helpers that absorb xdart's enumeration logic
verbatim (Qt-free, image-free):

- `_scan_manifest(source)` → `list | None`: `getattr(source, "scan_manifest", None)` probe
  (Protocol is `runtime_checkable` on names; `Scan` and duck sources without the method
  still work).
- `_incidence_extremes(manifest, motor)` → `(status, indices)`: the decision tree from
  `_gi_whole_scan_scout_entries` (`image_wrangler_thread.py:1278-1356`) **minus** image
  loading. It must preserve all current outcomes (verification fix): `float(motor)`
  succeeds → `("skip", [])`; `len(manifest) < 2` → `("skip", [])`; `< 2` *readable*
  incidences → `("unverifiable", [])`; `lo == hi` (single distinct incidence) → `("skip",
  [])`; else extremes selected **by resolved incidence value, never positional** →
  `("found", [lo_idx, hi_idx])`.
- `_resolve_incidence(meta, motor)` → `float | None`: verbatim port of
  `_resolve_incidence_from_meta` (`image_wrangler_thread.py:1359`).

`prepare_gi_freeze` **first** short-circuits to `status="skip"` when the plan ranges are
already pinned (`_gi_1d_freeze_key(plan) is None and not _gi_2d_freeze_keys(plan)`),
**before** calling `scan_manifest()` — this preserves the T0-3 silent skip
(verification fix; `test_gi_prepass_skips_scout_when_ranges_fully_pinned`,
`tests/xdart/test_gi_batch_real_data.py:1118`). On `"found"` it returns the plan with
`extra["gi_freeze_scout_indices"] = (lo, hi)`, `status="frozen"` (meaning *indices pinned*;
the range-freeze happens downstream), `scout_indices`, and the resolved `scout_refs`.
Otherwise it leaves `extra` unset with `status="skip"`/`"unverifiable"`.

**No `FrozenPlan` wrapper, no grid container.** The actual frozen ranges land in
`plan.integration_1d/2d` exactly as `_apply_gi_freeze_policy` already writes them — which is
what preserves the byte-compatible frozen v2 format. `prepare_gi_freeze` does no
integration, so the degenerate-scout `GIFreezeError` is raised by the *freeze* step, not by
prepare; prepare never raises for an unenumerable source.

### 4. The headless session consumes it with ZERO session changes

`prepare_gi_freeze` only writes `plan.extra`, which the session already consumes. A headless
whole-scan caller does:

```python
plan2, diag = prepare_gi_freeze(source, plan)
session = ReductionSession(plan2, source, gi_freeze_mode="scout_union")
```

`__post_init__` (`core.py:717`) builds the scan from the **whole** source; `_apply_gi_freeze`
→ `_apply_gi_freeze_policy` → `_gi_freeze_scout_indices` reads `extra["gi_freeze_scout_indices"]
= (lo, hi)` first (line 1697), loads exactly those two frames lazily (line 1616-1618), and
unions. Same `freeze_common_axis` math, byte-identical given the same extremes.

## Rationale

- **It relocates the one genuinely-misplaced thing — and only that.** Incidence-extent
  *discovery* is pure metadata logic; it has no business in a Qt thread. The freeze math is
  already headless and stays where it is. We move discovery, not the freeze invocation.
- **The seam is data, not code.** Discovery hands the session a `plan.extra` key it already
  reads. No new session method, no `ReductionSession` change, no new control flow.
- **Warn-and-proceed becomes a source capability, not a GUI decision.** "Can this scan be
  swept?" is `source.capabilities.has_scan_manifest` + `scan_manifest() is None`, computed
  headlessly. The GUI only *reacts* (emit an advisory, stamp provenance). This is exactly
  the reviewer's "source capability flag, not a GUI decision."
- **It is honest about scope.** One concrete function named for its only job. The reusable
  *seam* for a future `prepare_rsm_bounds()` is the `has_scan_manifest` capability + the
  `(status, indices)` scout pattern — not a framework shipped before its second user.
- **The equivalence spine stays byte-identical.** The frozen ranges are produced by the same
  union math over the same two extreme frames; prepare only relocates *which indices* are
  chosen. Given the same extremes, the spine cannot diverge.

## Alternatives considered

- **The full lifecycle framework** (`PrepareProvider` registry + `FrozenPlan` wrapper +
  auto-range as a second provider). **Rejected** — over-abstraction the guardrail forbids:
  the other claimed users are not prepare-stage today (auto-range is GUI state, common-mask
  does not exist, RSM scout is caller-wired, stitch is finalize). The function-per-pass shape
  graduates to a registry only when a second real user lands; the index-discovery helpers +
  the diagnostics type are the reusable nucleus if that day comes.
- **Pure `plan.extra` side-channel with no new type** (the minimal design). **Rejected** —
  too implicit for a cross-package contract: a stringly-typed dict key a caller can silently
  mismatch, and no structured carrier for the warn/abort/provenance disclosure xdart needs
  (today `scan.gi_freeze_diagnostic`). One tiny `PrepareDiagnostics` is worth it.
- **A two-flag capability** (`has_scan_manifest` + `extent_establishable`). **Rejected** —
  the second condition is a runtime scout outcome, not a static source property; it belongs
  in `PrepareDiagnostics`, not the capability.
- **prepare returns bare `scout_indices`, xdart re-derives paths.** **Rejected** — for a
  file series, frame-index ≠ file-number, so xdart would re-implement the enumeration it just
  deleted to map indices back to `(path, number, meta)`. `PrepareDiagnostics` carries the
  resolved `scout_refs` so the caller's loader is a plain `read_image` (verification fix).

## Consequences

- **What actually leaves xdart is incidence-extent *discovery* (~90 lines), not the freeze
  path.** Deleted from `image_wrangler_thread.py`: `_enumerate_scan_files` (:1373),
  `_resolve_incidence_from_meta` (:1359), and the extreme-finding half of
  `_gi_whole_scan_scout_entries` (:1278-1341). **Honest framing for the PR:** this resolves
  codex-P1 (whole-scan extent) but does **not** unify the freeze invocation. The
  freeze-over-the-extremes still runs in xdart's `freeze_live_scan_gi_ranges` throwaway
  session, because the live-streaming chunk-1 session genuinely cannot see frame `hi`.
- **What stays in xdart (correctly — Qt/thread concerns):** `freeze_live_scan_gi_ranges` +
  `_freeze_gi_1d/2d_auto_range` (the thin freeze invokers and their `_build_scout` /
  `_scout_pending_frames` helpers); `_warn_gi_first_chunk_freeze` (emits `showLabel`, stamps
  `scan.gi_freeze_diagnostic`); `_abort_gi_prepass` (sets `command='stop'` under
  `command_lock`); `_gi_ranges_fully_pinned` as a cheap GUI-dict pre-core latch;
  `_gi_freeze_whole_scan_prepass` shrinks to an orchestrator that calls
  `prepare_gi_freeze`, maps `diag.status` to the GUI reaction, and on `"frozen"` loads the
  two `scout_refs` and runs the existing freeze. The error catch around the freeze step
  **stays `except Exception`** (verification fix — narrowing to `except GIFreezeError` would
  let a scout-image read/PONI/mask error escape the worker thread, which has no top-level
  except: *worse* than the abort it is meant to preserve).
- **Two disclosure carriers during the transition:** `PrepareDiagnostics` (new) and
  `scan.gi_freeze_diagnostic` (existing byte-compat provenance string). xdart bridges them.
  Mild debt until the provenance string is itself sourced from the diagnostics object.
- **The equivalence spine is preserved but is NOT the gate for this move.** The spine
  (`tests/xdart/test_gi_batch_real_data.py::test_*_publication_live_batch_reload_equivalence`)
  hands both legs the same pre-frozen grid and is **index-set-invariant by construction** —
  it cannot detect a wrong scout index. The actual gates are the non-spine tests:
  `test_gi_streaming_prepass_scouts_whole_scan_extremes` (:956, `nums == [1, 5]` :986),
  `test_gi_union_scout_covers_all_frames_not_just_frame0` (:906),
  `test_gi_prepass_warns_and_proceeds_on_unestablishable_range` (:1014, currently asserts
  `status == "abort"` :1042 — must change to the new status enum), and
  `test_gi_prepass_warns_and_proceeds_on_image_directory_source` (:1053, `"Image Directory"
  in emitted[-1]` :1088). **"Keep the suite green frame-for-frame" is overstated:** these
  four tests are re-pointed at the new core entry points; the *numeric* freeze output is
  byte-identical, but the test surfaces change.
- **HDF5 single-writer / byte-compat v2 / ADR-0003 unaffected.** `prepare_gi_freeze` +
  `scan_manifest()` are read-only metadata; the throwaway freeze session has no write sink;
  frozen values land in `plan.integration_1d/2d`; nothing touches the event/record structure.

## Residual risk (honest)

1. **Eiger conservatism is preserved but not improved.** Eiger masters stay
   `has_scan_manifest=False` (per-frame incidence is in the SPEC sidecar, not cheaply
   enumerable from the master), so angle-dependent Eiger scans — exactly where incidence
   varies most — still warn-and-proceed on a first-chunk freeze. The GI grid policy
   (AUTHORITATIVE memory) accepts this. Solving it needs a SPEC-aware source with a real
   `scan_manifest()`; out of scope, but the seam now exists for it.
2. **The Step-2 `_frame_source_for(scan)` factory is the dominant regression risk.** xdart's
   batch path does not build a `FrameSource` today; it builds pending tuples + `LiveFrame`s.
   The factory must reproduce the strict `^{scan_name}_\d+\.{ext}$` discovery regex
   (`image_wrangler_thread.py:1390-1391`) — **not** `TiffSeriesSource.from_directory`, whose
   unanchored `fnmatch` glob (`sources/image.py:120`) would ingest neighbour files like
   `{scan}_again_0001.tif` ⇒ wrong extremes ⇒ silent clip — and set `has_scan_manifest` from
   the same `inp_type`/extension checks that gate `_enumerate_scan_files` returning `[]`.
3. **Frame-index vs file-number index space.** `TiffSeriesSource.frame_indices` is positional
   `range(1, len+1)` (`sources/image.py:96`); xdart's `_get_scan_info` keys on filename
   number. They coincide for contiguous 1-based scans (the spine's `Combi4` fixture), so the
   fail-loud membership check (`core.py:1703`) passes — but a Step-2 factory that mixes the
   two index spaces would trip it. The headless `scan_manifest()` default keys on
   `frame_indices`, so the headless path is self-consistent; the risk is confined to the
   factory.

## Status note for the maintainer

This is additive and reversible. **Step 1 (below) ships dead-but-proven core code** — xdart
still runs its own prepass, the spine stays green untouched — so it can land on `dev` without
a live checkpoint. **Step 2 (the xdart rewiring + deletion) is the regression-prone part and
requires a live beamline confirmation** (Stabilization C per `CLAUDE.md`) before merge,
because the `_frame_source_for` factory is new behavior on the live batch path. If the lift
proves not worth it, Step 1 leaves a clean, tested core API and no xdart change to revert.
**At release, the `ssrl_xrd_tools>=` floor must already cover this once Step 1 lands** (the
writer hard-imports core; see CLAUDE.md).
