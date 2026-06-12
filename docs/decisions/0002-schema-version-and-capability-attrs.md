# ADR-0002: integer schema version + per-feature capability attributes

**Status:** accepted · 2026-06-12 · (greenfield Difference 5)

## Context

The processed-scan schema stamps `ssrl_schema_version = 2`, but "version
2" now covers several generations of additive features (per-frame
geometry, the `frames/` record, `@source_base`, optional `sigma`,
`two_d_kind`).  The greenfield design asked for "2.1-style" additive
evolution.  Two mechanisms were on the table: fractional/minor version
numbers, or per-feature capability attributes.

## Decision

Keep the **integer** `ssrl_schema_version` and evolve via **per-feature
capability attributes** declared in `xrd_tools.io.schema` (a capability
registry with name, location, and introduced-version metadata; Phase 2f of
the implementation plan).

- The integer bumps only on a **breaking layout change** — which the
  frozen-format policy says should ideally never happen.  Readers warn
  (never refuse) on a newer integer (`warn_if_newer_schema`).
- A reader **feature-detects**: a capability is used iff its attribute is
  present AND the schema registry knows it.  Absence is silent and valid —
  every capability is optional by construction.
- Writers stamp capability attrs additively; old readers ignore unknown
  attrs by design.

## Rationale

- Fractional versions impose a total order on features that don't have
  one (a file can have geometry but no thumbnails, or vice versa).
  Capability attrs describe exactly what is present.
- Presence-based detection already matches how the readers behave; the
  registry turns that from convention into declared schema.
- The persisted attribute *keys* (`ssrl_schema`, `ssrl_schema_version`,
  `ssrl_dtype`) are frozen regardless — see the pins in
  `tests/core/test_schema_as_code.py`.

## Consequences

- `PROCESSED_SCHEMA_VERSION` stays `2` for the foreseeable future.
- New optional features ship as: schema-registry entry + capability attr
  stamped by the writer + feature-detecting reader + byte-compat-gated
  tests.  No version negotiation logic anywhere.
