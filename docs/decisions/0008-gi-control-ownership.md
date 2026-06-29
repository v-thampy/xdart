# ADR-0008: GI control ownership follows the instrument-vs-plan split

**Status:** accepted · 2026-06-27

## Context

The GUI three-section layout separates experimental configuration from processing
options. Grazing-incidence controls had drifted across those sections in the
design docs: some text put all GI controls in the experimental section, while
other text treated GI modes and output axes as integration-plan choices.

That ambiguity matters because reload, reintegrate, Stitch, and RSM all need the
same state split. The section that owns a value determines whether it is persisted
as part of the instrument/sample configuration or as provenance for one reduction
or analysis run.

## Decision

Use the hybrid split:

- **Experimental config owns sample facts**: measurement mode, sample material,
  incidence-angle source, fixed/manual incidence value, sample orientation, tilt,
  UB/sample state, and beam energy/wavelength provenance.
- **Processing options own run choices**: GI output axes/submodes, 1D/2D mode,
  q/chi/qip/qoop/exit-angle selections, `npts_oop`, integration ranges, bins, and
  which correction toggles are applied for this run.

The GUI may present these as adjacent controls, but the data model keeps the
instrument/sample facts separate from the per-run plan.

## Consequences

- Loading a `.nxs` hydrates the experimental GI facts into section 2 and hydrates
  the chosen output modes/ranges into section 3.
- Section 3 re-renders when section 2 changes measurement mode, because available
  axes and correction toggles depend on standard-vs-GI state.
- `GISettings` remains the shared sample/measurement object; reduction-specific
  `GIMode`/axis choices remain plan state until a later convergence is justified
  by code, not by GUI layout alone.
