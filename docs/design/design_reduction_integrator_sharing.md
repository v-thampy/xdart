# Design (SCOPED, NOT BUILT) — share the reduction integrator across workers

**Status:** deferred follow-up to MEM-3. The MEM-3 guardrail (cap the pool at the
throughput knee) captures the safe win now; this doc scopes the deeper win and
records why it is *not* being taken in the pre-tag window.

## Problem

The streaming reduction runs N worker threads. pyFAI `AzimuthalIntegrator`s are
not thread-safe — they lazily compute and cache per-pixel arrays on `self`
during integration (`calc_cartesian_positions`, the CSR engine, corner arrays),
so concurrent use on one object races (crashes in `calc_cartesian_positions`).
`_ReductionIntegratorProvider` works around this by **`copy.deepcopy`-ing the
whole integrator per worker thread** (`reduction/core.py`), which copies the
full geometry **and** the CSR LUT.

For the reference 2167×2070 (4.48M-pixel) Eiger at npt 500×500, one integrator
is ~0.8 GB, and the per-worker in-flight scratch adds the rest, so peak RSS
scales ~1 GB per worker:

| workers | peak RSS | 651-frame time |
|---|---|---|
| 2 | 6.0 GB | 36.5 s |
| **4 (knee)** | **9.1 GB** | **25.0 s** |
| 8 | 13.7 GB | 26.1 s |
| 16 | 19.3 GB | 26.8 s |

At the 4-worker knee (production default), ~3–4 GB of the ~9 GB floor is
duplicated integrator geometry. **MEM-3 caps the pool at the knee** (no point
paying 14–19 GB for 8/16 workers that add no throughput), but it does not lower
the knee's own floor. This doc is the only lever that does — at full speed.

## Proposed change

Share the read-only geometry/LUT across workers; give each worker only the
**mutable scratch** it needs. Two candidate shapes:

1. **Pre-warm + share the AzimuthalIntegrator.** Trigger every lazy cache once,
   single-threaded, at session open (compute `ttha/chia/corner arrays`, build
   the CSR engine for `(method, npt, unit)`), then hand the SAME integrator to
   every worker with `safe=False` so `integrate2d` never recomputes/mutates.
   Win: ~0.8 GB → shared once; workers hold only output buffers.
2. **Call the CSR engine directly.** Bypass the `AzimuthalIntegrator` wrapper:
   build the sparse `CsrIntegrator` engine once (shared, read-only LUT) and call
   it per frame with a per-call `(data, variance)` → `(I, sigma)`, so there is
   no shared mutable object at all. Cleaner isolation; more integration-layer
   surgery.

## Costs BEYOND coding complexity (why it is deferred)

- **Thread-safety correctness risk — the reason the deepcopy exists.** Sharing
  is only safe if *every* array pyFAI touches during `integrate2d` is already
  materialized and never re-derived. Any missed lazy path → a data race →
  crash or silent wrong output. Requires an exhaustive pre-warm + a concurrency
  stress test, and stays coupled to pyFAI internals.
- **Equivalence-spine risk (a guardrail).** The deepcopy is byte-identical to
  the base, which is what keeps `live≡batch≡reload` green. A shared engine that
  leaks any per-thread state could make results differ frame-to-frame → breaks
  the acceptance spine. Verification MUST include the full spine byte-compat at
  1/4/16 workers (same gate MEM-3 uses).
- **`safe=False` assumption.** Option 1 needs `safe=False` to avoid the per-call
  mask-CRC recompute racing. That assumes the mask is constant for the scan. The
  per-frame intensity threshold writes NaN into the *data*, not the mask, so it
  is *probably* fine — but that must be proven, not assumed.
- **pyFAI-version fragility.** Deepcopy is robust to pyFAI internals; "share the
  read-only bits" depends on which arrays a given pyFAI version treats as
  mutable, and can break silently on upgrade. Needs a pinned-version note + a
  regression test that fails loudly if pyFAI starts mutating post-warm.
- **Partial win.** Output/variance scratch stays per-worker (small), so it is
  ~0.8 GB → ~0.1 GB/worker, not zero.

## Decision

**Deferred.** MEM-3's cap removes the pure waste (8/16-worker blowup) safely and
today. This deeper fix trades a robust, boring memory cost for an ongoing
correctness/maintenance risk on the equivalence-critical path, for a ~3 GB win
at the 4-worker default. Schedule it as a dedicated post-tag chunk with: an
exhaustive lazy-cache pre-warm, a concurrency stress test, the full spine
byte-compat at 1/4/16, and a pyFAI-mutation regression guard. Alternative
cheaper lever for the same floor: finding [5] (kill the float64 ingest upcast)
shrinks the in-flight scratch half of the per-worker cost with none of these
risks.
