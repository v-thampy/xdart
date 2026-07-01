# ADR-0009: calibration wavelength is the canonical energy source

**Status:** accepted · 2026-06-27

## Context

Energy can appear in several places: a calibration/PONI wavelength, an
`RSMPlan.energy`, and `GICorrectionStack.energy_eV`. The docs previously contained
both a resolved statement and a “decision needed” note, which made it unclear
which value a GUI energy widget should own.

## Decision

The calibration wavelength persisted under `/entry/diffractometer` is the source
of truth for beam energy/wavelength when it is present. Other energy fields are
derived from it or validated against it:

- `RSMPlan.energy` derives from the calibration wavelength unless the caller
  supplies an explicit override for a source that has no authoritative persisted
  wavelength.
- `GICorrectionStack.energy_eV` is bound to the same section-2 beam value in the
  GUI and is checked against the calibration wavelength in headless runs.
- Divergence is loud: the implementation should warn or fail according to the
  existing consistency guard rather than silently mixing energies.

## Consequences

- Section 2 has one beam-energy display/edit point.
- GI material remains sample state, while GI correction energy is a bound value,
  not a second independent user input.
- RSM/Stitch/Int notebooks should read “load calibration/source energy → build
  plan using derived energy” rather than asking users to duplicate energy in every
  plan.
