# ADR-0005: the authoritative FrameRecord store lives in the session; the GUI holds only derived projections

**Status:** accepted · 2026-06-13 · (greenfield Difference 2 + Difference 3; refines ADR-0004 §4)
**Builds on:** ADR-0003 (record cardinality), ADR-0004 (event/threading/flush contract).

## Context

ADR-0003 fixed the record *shape* — `Mapping[frame_index, FrameRecord]`, each record
a multi-result fold of single-result events — but deliberately not *where that map
lives*. ADR-0004 then parked the save cadence (`FlushPolicy`) + eviction in the xdart
adapter, justified by "the in-memory eviction pressure only exists in the GUI cache."

A reviewer caught the hidden coupling: that justification is true *today* only because
the authoritative store is GUI-side, and it **silently assumes it stays there**. If the
store stays in xdart, then Difference-2's headline goal — *"xdart has no API to own
data"* — is approached but never actually **reached**: xdart would still own the
canonical `FrameRecord` store, with the GUI's three hand-synced caches (`data_1d`,
`data_2d`, `PublicationStore`) as the authoritative source (deep-review v2 #1's "three
stores hand-synced on the live frame path" hazard). So the cadence-location decision in
ADR-0004 and the store-ownership decision are **the same decision, half-made**. Phase 5
(`FrameRecord` + the one-store collapse) will bake the answer in either way — so, per the
ADR-0003 discipline, make it on purpose, before Phase 5 freezes it.

## Decision

**The single authoritative `Mapping[frame_index, FrameRecord]` store lives in the
headless session layer (`xrd_tools.session`), not in xdart.** Concretely:

- The session owns a **bounded record store**: persist-before-evict (an unsaved record
  is never evicted), and **hydration of an evicted record from disk via the headless
  reader API** (`read_frame_view` / `read_scan`) — all Qt-free. This is the headless
  successor to xdart's `LiveFrameSeries` + the parallel `data_1d`/`data_2d` dicts, folded
  into one store keyed by frame index (ADR-0003).
- **The save cadence + eviction bound follow the store into the session.** The
  `FlushPolicy` decision and the persist-before-evict bound are session concerns — this
  **refines ADR-0004 §4**: its "cadence stays an xdart concern" was correct *only* while
  the store was GUI-side; once the bounded store (with its eviction pressure) is the
  session's, the cadence that serves that pressure is too.
- **What stays in the xdart bridge:** the *Qt/file-handle-bound* pieces only — the
  h5pool-bracketed `_save_to_nexus` flush **action** (the writer thread + file lock are
  xdart's), the `QueuedConnection` event marshaling, and the **derived, display-only
  projections**: the thumbnail tier, the bounded raw-image window for the 2D panel, the
  publication verdict. These are rebuilt from the event stream and are **never
  authoritative** — losing them costs only a re-render, not data.
- So the GUI becomes a **read-only projection of the session's event stream**, which is
  what makes "thin xdart" structural (a property of construction) rather than a
  discipline maintained by hand-syncing.

## Rationale

- **It makes Difference 2 real, not aspirational.** "xdart owns no data" is enforced by
  where the authoritative store is constructed, not by review vigilance.
- **It dissolves the three-store drift hazard (v2 #1) by construction.** One authoritative
  record; the cake/1D/raw/metadata panels read *projections* of it, so they cannot
  disagree about the same frame — the live≡reload invariant ADR-0003 keyed by
  `(frame, mode)` becomes a fold the projections can't fork.
- **It is reusable.** A headless live monitor, a notebook, or a remote/Tiled client gets
  the same bounded store + hydration without the GUI — the session is the service.
- **The cadence-follows-store coupling is honest.** Eviction pressure is a property of a
  bounded store; putting the store and its cadence in the same layer keeps them from
  drifting (the exact ADR-0004 footgun, now resolved deliberately).

## Alternatives considered

- **Store stays GUI-side (status-quo direction ADR-0004 implied).** Rejected: it caps
  Difference 2 at "approached," keeps the three-store hand-sync, and makes any future
  headless consumer reimplement the store. The only thing it saves is building a headless
  bounded store — which Phase 5 builds regardless.
- **Split: authoritative results in the session, but ALL caching (incl. thumbnails/raw)
  in the session too.** Rejected as over-reach: thumbnails + the raw 2D window are
  *display* artifacts (sized for the screen, cheap to recompute), genuinely a GUI concern;
  forcing them headless couples the session to display policy. The clean line is
  **authoritative record → session; derived display projections → GUI.**

## Consequences

- **Phase 5 builds the bounded `FrameRecord` store inside the session** (persist-before-
  evict + disk hydration), and `data_1d`/`data_2d`/`LiveFrameSeries`/`PublicationStore`
  collapse into it (authoritative) + GUI projections (derived) — the one-store move
  (Difference 3 / X1) happens here, in one step, on the ADR-0003 record shape.
- **`FlushPolicy` + the eviction bound move from the xdart adapter into the session**
  (refines ADR-0004 §4); the h5pool bracket + Qt marshaling remain the xdart bridge's.
- **The 4f-bridge** (rewiring `ScanSessionAdapter`) should be built knowing the store is
  destined for the session — i.e. the bridge maps events to GUI *projections*, and does
  not establish xdart as the long-term record owner.
- **Acceptance gate addition (Phase 5):** because the multi-result reload path is brand
  new (per-mode `results_1d`/`results_2d` maps written + read), the equivalence spine must
  grow a **multi-mode case** — store ≥2 GI sub-modes → reload → each mode byte-identical —
  or the most schema-touching change of the cycle ships without the gate that protects
  everything else (reviewer note 2; recorded in `phase4_scansession_design.md` §5).

## Status note for the maintainer

This is the second keystone (after ADR-0003) and it reframes part of ADR-0004. Like
ADR-0003 it is **reversible**: nothing is built against it yet (Phase 5 has not started),
so the store could still be kept GUI-side if the headless-store lift proves not worth it.
But it is the decision that determines whether "thin xdart" is structural or aspirational,
so it is recorded *before* Phase 5 bakes one answer in. Lean strongly toward
session-ownership for the reasons above.
