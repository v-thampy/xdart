# Design: GI controls → section 2/3 split + 2-way load hydration (Jun 19, 2026)

**Status:** DESIGN/P7, reconciled to
[`ADR-0008`](../decisions/0008-gi-control-ownership.md) and
[`ADR-0009`](../decisions/0009-energy-single-source.md).

Branch: `panel-sections-refactor`.  Builds on the panel-sections refactor (shared
`StaticControls`, integrator-owned threshold row, Browse-inline) and on the
reintegrate-on-reload round-trip fixes (calibration, `sample_orientation`/`tilt`
via `gi_config` fallback, mask via `detector_shape`).

## Problem / motivation

Reintegrate-on-reload kept diverging from the live reduction because the
**reloaded scan** and the **GUI controls** are two separate states, and the
reintegrate plan reads a mix of the two:

- Some GI params round-trip via the `.nxs` (`gi_config`, `geometry`, mask,
  calibration, per-frame `incident_angle`).
- Others are driven only by the live GUI controls and are **not** hydrated from
  the `.nxs` on load: the **`gi` flag** (wrangler "Grazing Incidence" checkbox →
  `scan.gi` via `sigUpdateGI`/`update_scattering_geometry`), the **incidence
  motor**, and the **gi_mode** combos.  Reload + mismatched controls →
  reintegrate uses the wrong geometry/mode.

(`skip_2d` is NOT in this class — reintegrate passes `integrate_2d` explicitly
via the Reintegrate 1D/2D buttons.)

**Goal:** one canonical home for every GI value, with reload hydrating the visible
controls. ADR-0008 splits ownership: sample/measurement facts live in section 2,
while output axes, GI submodes, bins, ranges, and correction toggles live in
section 3. The GUI may place the controls adjacently for usability, but saved
state and plan provenance follow that split.

## The incidence-motor separation (the part that made this feel cursed)

The scary "incident angle comes from the metadata file" is a **live-acquisition**
concern, not a reintegrate one:

- The per-frame incident **angle** is baked into the `.nxs`
  (`per_frame_geometry/incident_angle`) and used by `_resolve_gi_incident_angle`.
  Reintegrate uses it directly and **never re-reads the metadata file**.
- The motor **name** is only needed when acquiring/reducing fresh (which SPEC
  column to read).

So: **selection** of the motor is a section-2 sample/measurement fact; the
**metadata read** stays in the wrangler thread (live).  On a loaded `.nxs` the
motor box is informational (angle already baked).

## Plan (stage-by-stage, live-gated after each)

### Stage A — GI section split (controls + ownership)
- Section 2c: GI on/off toggle, Sample Orientation, Tilt, Theta Motor dropdown,
  Theta Value (manual), material, and incidence source.
- Section 3: GI *mode* combos (`axis1D`/`axis2D`), `npts_oop`, ranges, bins, and
  correction toggles.
- `get_gi_config()` reads the section-2 facts → dict; session-persist; reveal
  motor/orientation/tilt/material only when GI is on. The GI toggle drives
  `scan.gi` through the existing `update_scattering_geometry`/`sigUpdateGI` seam.
- Live injection: before `wrangler.setup()`, push the integrator GI config into
  the wrangler params — same seam as the threshold row
  (`_push_threshold_to_wrangler`).  So **live == panel**.
- The GUI writes section-2 GI facts into `scan.gi_config` and writes section-3
  run choices into `scan.bai_*_args` so **reintegrate == panel**.

### Stage B — motor wire + default order
- wrangler → integrator: when metadata loads (`set_gi_motor_options`), emit the
  SPEC motor-column list; the integrator dropdown populates (`th`/`Manual` always
  + discovered columns).
- Default-select order when `th` absent (case-insensitive): `th`, `theta`, `eta`,
  `halpha`, `gth`, `gonth`, else first available / `Manual`.
- integrator → wrangler: selected motor injected at run setup (Stage A seam).
- **Depends on** the metadata reader capturing all motors (deferred item F6 in
  `CC_preship_sweep_deferred_jun2026.md`) for the full non-standard-motor list;
  ships functional with `th`/`Manual` until then.

### Stage C — load hydration (the 2-way sync)
- On `.nxs` load, section 2 hydrates from `gi_config`/diffractometer/beam state
  and section 3 hydrates from `bai_1d_args` + `bai_2d_args` (units, npts 1D/2D,
  ranges, gi modes). Signals are **BLOCKED** so hydration cannot trigger a
  spurious reintegrate or session churn. Hydrating GI on/off sets `scan.gi` to
  match the loaded scan, closing the footgun.

### Stage D — cleanup + tests
- Hide/remove the wrangler GI group once section 2 owns the sample facts; keep
  only the motor-options provider.
- Tests: load → panel reflects saved params; GI hydration round-trip; motor
  default-order (pure unit test); live≡batch≡reload equivalence spine + full
  suite green.

## Guards / risks
- Block signals during hydration (no spurious reintegrate / session churn).
- GI-checkbox ownership transfer (wrangler → integrator) must keep `scan.gi`
  correct on every path.
- Motor options arrive async (metadata load).
- Keep the `gi_config`/`detector_shape` reintegrate fallbacks as
  belt-and-suspenders.
- Live-gated after each stage (quit & relaunch `xdart`).
