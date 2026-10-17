# Limitations and workarounds

These notes apply to the 1.2.0 release being prepared. They describe accepted
behavior and remaining limits; they do not certify a platform or announce a
published release.

## Reprocess outputs made before the seam-q correction

Older **Standard radial q/2θ** reductions could truncate the requested azimuthal
region when translating a χ window across pyFAI's ±180° discontinuity. Affected
profiles can therefore contain different intensities from a corrected run.

**Use Replace (shown as Overwrite in some controls) to regenerate the complete
affected output from the original raw data. Do not use Append for this
migration.** Old and corrected seam-window frames can have the same recorded
configuration, so Append may keep old truncated frames and add corrected ones
without detecting that mixture. An update does not repair existing files.
Keep a copy of the old output if it is needed for comparison.

Historical angular-support reconstruction can report insufficient information
when the actual admitted directions cannot be established. Matching q bins do
not prove that an older result covered the same detector directions. The
correction is bounded to Standard radial integration; GI and Stitch have
separate window semantics.

## Explicit full-turn I(χ) and cake ranges: use Auto

An explicit full-turn χ interval combined with a χ offset remains a limitation
for **I(χ) profiles and 2D cakes**. The known case is a typed range of
`−180°…180°` with an offset of `30°`: the I(χ) result can be all NaN with a
collapsed axis, and strict cake integration refuses the result. This is the
case called **W1** in the regression tests; Standard radial q is corrected.

For full-turn I(χ) or cake output, select **Auto for the azimuth/χ range** instead
of typing the explicit endpoints. Keep the intended χ offset. This Auto is the
range setting, not the detector-intensity Threshold control. Arbitrary shifted,
wider-than-one-turn or zero-width χ intervals are not newly qualified by this
release; use a conventional supported range and check the saved axes.

## Exact full-turn Standard radial ranges use Auto admission

For **Standard radial q/2θ** integration, an explicit interval exactly 360° wide
is treated as Auto, including when χ offset is zero. This includes every
azimuthal direction and avoids endpoint wrapping. It does not preserve a
distinct endpoint-clipping interpretation for a typed full turn.

This conversion does not extend the correction to I(χ), 2D cakes, GI or Stitch.
Use the Auto workaround above for full-turn I(χ)/cake output.

## Seam-crossing windows can use additional memory

A corrected seam-crossing partial window can require an alternate integrator
for each worker and its current Standard integrator. That integrator retains
additional detector/centre state. Its extra memory cost has not been quantified;
the correction does not establish a new memory bound or disposal guarantee.

If memory is tight, reduce **Cores** and monitor the process during a
representative scan. The application's memory target is advisory, so a warning
is not a promise that the run will stay under that target.

## Supported GUI installation routes

Use **Pixi**, the Pixi-based installer, or **conda/mamba** with the compiled
dependencies from conda-forge. Those routes supply xrayutilities and its paired
NumPy rather than relying on a local compiler or mixing incompatible runtimes.
Follow the [installation guide](INSTALLATION.md) for the complete commands.

A pip-only `xdart[gui]` or `uv tool install "xdart[gui]"` installation is
**best-effort**, including native dependency builds and compatibility. The
headless `pip install xdart` route is unaffected by this GUI support policy.

## Platform and output boundaries

Native Windows/SMB finite-result behavior and native GUI interaction need their
own qualification. A passing offscreen or hosted test does not establish those
results. Preserve raw inputs when moving projects: saved scientific results can
reopen independently, while full raw-image inspection and processing replay
still depend on the referenced sources.

SVG plot export uses a compatibility fix for pyqtgraph 0.14.0 and Qt 6.11.
Saved Stitch GI cut XYE companions use `sqrt(abs(I))` as an
estimate in the third column, as their headers state; it is not a propagated
uncertainty. Exporting an arbitrary displayed cut is a separate operation from
the automatically written companions.
