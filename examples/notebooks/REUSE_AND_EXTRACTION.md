# Notebook Reuse and Extraction Record

## Scientific Operations

| Repeated operation | Previous copies | Shared owner/API | Disposition and direct coverage |
| --- | --- | --- | --- |
| Processed 1-D scan selection and natural ordering | flash notebook, processed-data viewers, manual glob loops | `analysis.discover_processed_scans`, `load_time_resolved_series` | Standardized; tests reject raw/unrelated NeXus files and retain scan-qualified labels. |
| Lazy frame/cake/raw/thumbnail lookup | processed viewers, flash notebook | `TimeResolvedSeries`, `io.get_1d/get_2d/get_raw_frame/get_thumbnail` | Standardized; raw has no thumbnail fallback and tests cover qualified lazy access. |
| Monitor and reference-band normalization | flash/HZO notebooks | `normalize_monitor`, `normalize_reference_band` | Standardized with validity masks and provenance; zero/missing monitor values remain invalid rows. |
| Whole-pattern outlier flags | flash notebook | `flag_normalization_outliers` | Standardized robust factor/MAD policy; never pointwise-clips peaks. |
| Scan-safe temporal binning and time zero | flash notebook | `bin_time_resolved`, `select_time_zero` | Standardized; tests retain source frame ranges/counts and prohibit cross-scan bins. |
| Pilot/batch peak fitting and quality columns | HZO and flash notebooks | `PeakFitPlan`, `PeakFitAnalyzer`, `fit_peak_series`, `flag_fit_quality` | Existing plan/runner reused; runner now projects redchi/chisqr/AIC/BIC for compact results. |
| q/HKL lattice, calibration, rate, export | flash notebook | `add_lattice_results`, calibration classes, `add_temperature_results`, `export_time_resolved_results` | Standardized with uncertainty propagation, physical-time guard, NetCDF-safe attrs, and CSV tests. |
| Static pattern/image plotting | every viewer notebook | `viz.plot_1d`, `viz.plot_image` | Existing public helpers reused. |
| Interactive waterfall, saved-fit, thermal trends | flash notebook | `plot_waterfall`, `plot_peak_fit_frame`, `plot_thermal_history` | General xarray contracts promoted to `viz.plotly` and tested without mutating inputs. |
| Stitch and RSM gridding | legacy Stitch/RSM notebooks | `StitchPlan/run_stitch`, `RSMPlan/run_rsm`, `RSMVolume` | Existing headless owners reused; experiment geometry remains configuration. |

Deferred: sustained raw-source cursors belong to R2/source ownership; writer/session
work belongs to its active lane; beamline-specific calibration and thermal models
remain experiment configuration rather than a generic library conversion.

## Widget and Plot Reuse

| Surface | Decision | Rationale |
| --- | --- | --- |
| `PatternViewer` | Reused unchanged in 03, 07, 08 and staged 1-D/fit/stitch notebooks | It already accepts public patterns and offers log/offset controls without rerunning analysis. |
| `ImageViewer` | Reused unchanged in 07, 08 and staged 2-D viewer | It supplies display-only log/percentile controls and never stores a detector stack. |
| `PeakFitControls` | Reused in 03 | Its explicit Fit button and non-continuous sliders match pilot fitting. |
| `PhaseFitControls`, `PhaseFitViewer`, `BatchPhaseFitViewer` | Audited; retained for real CIF-backed phase workflows | The smoke examples lack meaningful phase/CIF configuration, so creating synthetic phase UI would be misleading. |
| Matplotlib helpers | Reused in 01, 02, 09 | Compact static validation plots remain appropriate. |
| Plotly helpers | Reused/extended in 08 | Waterfall, saved-fit, and thermal history have general xarray contracts and belong in `viz.plotly`. |
| Notebook-local controls | Plain ipywidgets composition only | File/configuration, status, export opt-in, and button wiring are workflow-specific; callbacks call headless APIs and expensive work is explicit. |
