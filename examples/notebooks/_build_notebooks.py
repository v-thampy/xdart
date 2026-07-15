"""Build the clean, deterministic public xrd_tools notebook examples.

The generated notebooks intentionally contain no outputs or execution counts.
They default to small synthetic inputs so CI and first-time users can execute
them without beamline data; each configuration cell provides the explicit
switch and paths for real data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parent


def md(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def notebook(cells: list):
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    for index, cell in enumerate(nb.cells):
        identity = f"{index}\0{cell.cell_type}\0{cell.source}".encode()
        cell.id = hashlib.sha256(identity).hexdigest()[:8]
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return nb


def write(name: str, cells: list) -> None:
    nbformat.write(notebook(cells), ROOT / name, version=4)


CONFIG = code(
    """
    import os
    from pathlib import Path

    # Smoke mode is deterministic and has no external-data requirement.
    # Set XDART_NOTEBOOK_SMOKE=0 and XDART_TEST_DATA=/path/to/data for real data.
    SMOKE_MODE = os.environ.get("XDART_NOTEBOOK_SMOKE", "1") != "0"
    TEST_DATA = Path(os.environ.get("XDART_TEST_DATA", "")).expanduser()
    """
)


NB_BATCH_INTEGRATION = [
    md(
        """
        # Batch Integration

        Low-level 1-D and 2-D integration reference. The smoke path makes a
        compact synthetic pattern; real mode uses an explicit image and PONI.
        For a durable multi-frame reduction, continue to notebook 06.
        """
    ),
    code(
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.core.containers import IntegrationResult1D
        from xrd_tools.integrate import integrate_1d, integrate_2d, load_poni
        from xrd_tools.io import read_image
        from xrd_tools.viz import plot_1d
        """
    ),
    CONFIG,
    code(
        """
        image_file = TEST_DATA / "image.tif"
        poni_file = TEST_DATA / "calibration.poni"
        npt_1d = 256
        npt_2d = (256, 72)
        run_button = widgets.Button(description="Run integration", button_style="primary")
        status = widgets.HTML("<i>Configure real paths, then run explicitly.</i>")
        display(widgets.VBox([widgets.HBox([run_button]), status]))
        """
    ),
    code(
        """
        if SMOKE_MODE:
            q = np.linspace(1.0, 4.0, npt_1d)
            intensity = 20 + 90 * np.exp(-0.5 * ((q - 2.35) / 0.07) ** 2)
            result_1d = IntegrationResult1D(q, intensity, unit="q_A^-1")
        else:
            assert image_file.is_file(), f"Missing detector image: {image_file}"
            assert poni_file.is_file(), f"Missing PONI calibration: {poni_file}"
            result_1d = integrate_1d(read_image(image_file), load_poni(poni_file), npt=npt_1d)

        fig, ax = plt.subplots(figsize=(7, 3))
        plot_1d(ax, result_1d.radial, result_1d.intensity, fmt="-", attrs={"xlabel": f"q ({result_1d.unit})", "ylabel": "Intensity"})
        plt.show()
        """
    ),
]


NB_STITCHING = [
    md(
        """
        # Multi-Geometry Stitching

        Combine angle-separated detector images with one calibrated integrator
        per image. Normalization is a per-image monitor factor, not a visual
        adjustment; inspect overlap and correction assumptions before analysis.
        """
    ),
    code(
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.integrate import create_multigeometry_integrators, stitch_1d, stitch_2d
        from xrd_tools.viz import plot_1d
        """
    ),
    CONFIG,
    code(
        """
        image_paths = []  # Real mode: ordered detector images.
        poni_paths = []   # Real mode: one PONI per image.
        monitor = None    # Optional one value per image.
        q_range = widgets.FloatRangeSlider(value=(1.0, 4.0), min=0.5, max=6.0, step=0.05, description="q range", continuous_update=False)
        compute = widgets.Button(description="Compute stitch", button_style="primary")
        display(widgets.VBox([q_range, compute]))
        """
    ),
    code(
        """
        q = np.linspace(1.0, 4.0, 220)
        left = 15 + 70 * np.exp(-0.5 * ((q - 2.15) / 0.10) ** 2)
        right = 15 + 70 * np.exp(-0.5 * ((q - 2.15) / 0.10) ** 2) * 1.02
        if not SMOKE_MODE:
            assert image_paths and len(image_paths) == len(poni_paths)
            integrators = create_multigeometry_integrators(poni_paths)
            stitched = stitch_1d(image_paths, integrators, radial_range=tuple(q_range.value), normalization=monitor)
            q, merged = stitched.radial, stitched.intensity
        else:
            merged = (left + right) / 2
        fig, ax = plt.subplots(figsize=(7, 3))
        plot_1d(ax, q, merged, fmt="-", attrs={"xlabel": "q (A^-1)", "ylabel": "Intensity", "title": "Stitched pattern"})
        plt.show()
        """
    ),
]


NB_PHASE_PEAK = [
    md(
        """
        # Phase and Peak Fitting

        Use the structure-agnostic `PeakFitPlan` path for a compact pilot fit.
        Structure-informed phase fitting remains a separate operation because it
        needs explicit CIF-backed phase models and scientific constraints.
        """
    ),
    code(
        """
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.analysis import AnalysisInput, PeakFitAnalyzer, PeakFitPlan
        from xrd_tools.gui.widgets import PatternViewer, PeakFitControls
        from xrd_tools.viz import plot_peak_fit
        """
    ),
    CONFIG,
    code(
        """
        q = np.linspace(2.3, 3.2, 320)
        intensity = 10 + 120 * np.exp(-0.5 * ((q - 2.76) / 0.018) ** 2)
        plan = PeakFitPlan(positions=(2.76,), model="gaussian", background="linear", sigma_init=0.02)
        controls = PeakFitControls()
        viewer = PatternViewer(patterns=[(q, intensity, "pilot")])
        display(widgets.VBox([controls.widget, viewer.widget]))
        """
    ),
    code(
        """
        outcome = PeakFitAnalyzer(plan).analyze(AnalysisInput(label="pilot", x=q, y=intensity, x_unit="q_A^-1"))
        assert outcome.ok, outcome.message
        figure = plot_peak_fit(q, intensity, outcome.result.payload, title="Pilot peak fit")
        figure
        """
    ),
]


NB_BATCH_PHASE = [
    md(
        """
        # Batch Phase Fitting

        The generic batch runner preserves one output row per input, including
        failures. The smoke path demonstrates that alignment with a peak plan;
        in real work replace the analyzer with `PhaseFitAnalyzer` and explicit
        CIF-derived phase models before running the button.
        """
    ),
    code(
        """
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.analysis import AnalysisInput, PeakFitAnalyzer, PeakFitPlan, batch_params_table, run_batch
        from xrd_tools.gui.widgets import BatchPhaseFitViewer, PhaseFitControls
        """
    ),
    CONFIG,
    code(
        """
        q = np.linspace(2.3, 3.2, 260)
        run_button = widgets.Button(description="Run batch fit", button_style="primary")
        quality = widgets.FloatSlider(value=0.02, min=0.005, max=0.1, step=0.005, description="width", continuous_update=False)
        status = widgets.HTML("<i>Smoke uses a bounded three-pattern batch.</i>")
        display(widgets.VBox([quality, widgets.HBox([run_button]), status]))
        """
    ),
    code(
        """
        patterns = [
            8 + 80 * np.exp(-0.5 * ((q - center) / quality.value) ** 2)
            for center in (2.755, 2.760, 2.765)
        ]
        analyzer = PeakFitAnalyzer(PeakFitPlan(positions=(2.76,), model="gaussian", background="linear", sigma_init=0.02))
        outcomes = run_batch(analyzer, [AnalysisInput(str(i), q, y) for i, y in enumerate(patterns)])
        labels, columns = batch_params_table(outcomes)
        assert len(labels) == len(patterns) == len(columns["center_0"])
        columns
        """
    ),
]


NB_SIN2PSI = [
    md(
        """
        # Grazing-Incidence sin2psi Strain Analysis

        Build a GI polar map, fit a reflection across chi sectors, then regress
        d-spacing against sin2(psi). Stress is only meaningful after choosing
        material-specific elastic constants and confirming the geometry.
        """
    ),
    code(
        """
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.analysis import Sin2PsiPlan, run_sin2psi
        from xrd_tools.core.containers import IntegrationResult2D
        from xrd_tools.gui.widgets import PeakFitControls
        """
    ),
    CONFIG,
    code(
        """
        q_range = widgets.FloatRangeSlider(value=(2.75, 3.25), min=2.0, max=4.0, step=0.01, description="q fit", continuous_update=False)
        fit_button = widgets.Button(description="Fit sectors", button_style="primary")
        display(widgets.VBox([q_range, fit_button]))
        """
    ),
    code(
        """
        q = np.linspace(2.5, 3.5, 260)
        chi = np.linspace(-35.0, 35.0, 25)
        intensity = np.empty((q.size, chi.size))
        for index, angle in enumerate(chi):
            center = 3.0 + 0.012 * np.sin(np.deg2rad(abs(angle))) ** 2
            intensity[:, index] = 20 + 300 * np.exp(-0.5 * ((q - center) / 0.025) ** 2)
        polar = IntegrationResult2D(q, chi, intensity, unit="q_A^-1", azimuthal_unit="chi_deg")
        result = run_sin2psi(Sin2PsiPlan(q_range=tuple(q_range.value), chi_width=7.0), polar).payload
        assert np.isfinite(result.d0)
        {"d0_A": result.d0, "slope_A": result.slope, "r_squared": result.r_squared}
        """
    ),
]


NB_REDUCTION = [
    md(
        """
        # Headless Reduction Pipeline

        `ReductionPlan` owns the scientific settings; `run_reduction` owns
        execution; a sink owns output. This is the recommended batch-reduction
        path and deliberately has no Qt dependency.
        """
    ),
    code(
        """
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.reduction import Integration1DPlan, Integration2DPlan, MemorySink, ReductionPlan, run_reduction
        """
    ),
    CONFIG,
    code(
        """
        plan = ReductionPlan(integration_1d=Integration1DPlan(npt=256), integration_2d=Integration2DPlan(npt_rad=128, npt_azim=48))
        run_button = widgets.Button(description="Run reduction", button_style="primary")
        chunk_size = widgets.BoundedIntText(value=4, min=1, max=64, description="chunk")
        progress = widgets.IntProgress(value=0, min=0, max=1, description="frames")
        display(widgets.VBox([chunk_size, run_button, progress]))
        """
    ),
    code(
        """
        # Real mode supplies a reduction.Scan with Frame image/source_path values and a PONI.
        # The smoke path validates the plan and sink without writing source data.
        sink = MemorySink()
        assert plan.integration_1d.npt == 256
        assert sink.frames == {}
        """
    ),
]


NB_READING = [
    md(
        """
        # Reading Processed NeXus

        Read compact 1-D/2-D products by frame label. A thumbnail is a stored
        display image; a full raw frame is loaded separately and may be absent
        after source data moves.
        """
    ),
    code(
        """
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.gui.widgets import ImageViewer, PatternViewer
        from xrd_tools.io import get_1d, get_2d, get_metadata, get_raw_frame, get_thumbnail, open_scan
        """
    ),
    CONFIG,
    code(
        """
        processed_file = TEST_DATA / "processed.nxs"
        frame_selector = widgets.BoundedIntText(value=0, min=0, max=999999, description="frame label")
        source_root = widgets.Text(value="", description="source root")
        inspect_button = widgets.Button(description="Inspect frame", button_style="primary")
        display(widgets.VBox([source_root, frame_selector, inspect_button]))
        """
    ),
    code(
        """
        if not SMOKE_MODE:
            assert processed_file.is_file(), f"Missing processed NeXus: {processed_file}"
            scan = open_scan(processed_file, source_root=source_root.value or None)
            selected = int(frame_selector.value)
            pattern = scan.get_1d(selected)
        else:
            # Smoke exercises the public viewers without fabricating a detector stack.
            q = np.linspace(1.0, 4.0, 180)
            pattern = type("Pattern", (), {"q": q, "intensity": 12 + np.sin(q) ** 2})()
            selected = 0
        pattern_viewer = PatternViewer(patterns=[(pattern.q, pattern.intensity, f"frame {selected}")])
        image_viewer = ImageViewer(np.arange(100, dtype=float).reshape(10, 10), title="Thumbnail or raw frame")
        display(widgets.Tab(children=[pattern_viewer.widget, image_viewer.widget]))
        """
    ),
]


NB_TIME_RESOLVED = [
    md(
        """
        # Time-Resolved XRD Analysis

        Load compact processed 1-D stacks into xarray, keep raw/cake rows lazy,
        normalize and bin with provenance, inspect a pilot fit, then derive a
        lattice and explicitly illustrative temperature history. Physical rates
        require explicit seconds and a calibration.
        """
    ),
    code(
        """
        import tempfile
        from pathlib import Path

        import h5py
        import numpy as np
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.analysis import (
            LinearThermalExpansion, PeakFitPlan, add_lattice_results,
            add_temperature_results, bin_time_resolved, fit_peak_series,
            export_time_resolved_results,
            flag_fit_quality, flag_normalization_outliers, load_time_resolved_series,
            normalize_monitor, normalize_reference_band, select_time_zero,
        )
        from xrd_tools.gui.widgets import ImageViewer, PatternViewer
        from xrd_tools.viz import plot_peak_fit_frame, plot_thermal_history, plot_time_resolved_waterfall
        """
    ),
    CONFIG,
    code(
        """
        processed_selection = TEST_DATA / "Pt_test_burst_00007.nxs"
        source_root = TEST_DATA if TEST_DATA.exists() else None
        q_band = widgets.FloatRangeSlider(value=(3.05, 3.20), min=2.4, max=3.3, step=0.01, description="reference q", continuous_update=False)
        bin_size = widgets.BoundedIntText(value=2, min=1, max=100, description="bin size")
        pilot_button = widgets.Button(description="Run pilot fit", button_style="primary")
        batch_button = widgets.Button(description="Run bounded batch", button_style="primary")
        export_enabled = widgets.Checkbox(value=False, description="Enable compact export")
        export_directory = Path(os.environ.get("XDART_NOTEBOOK_OUTPUT", tempfile.gettempdir()))
        status = widgets.HTML("<i>Buttons run expensive fits; display controls only redraw saved data.</i>")
        display(widgets.VBox([q_band, bin_size, widgets.HBox([pilot_button, batch_button]), export_enabled, status]))
        """
    ),
    code(
        """
        if SMOKE_MODE:
            smoke_file = Path(tempfile.gettempdir()) / "xdart_notebook_time_resolved_smoke.nxs"
            q = np.linspace(2.45, 3.30, 240)
            frames = np.arange(16, dtype=np.int64)
            centers = 2.765 - 0.0008 * frames
            stack = np.array([15 + 180 * np.exp(-0.5 * ((q - center) / 0.018) ** 2) for center in centers], dtype=np.float32)
            with h5py.File(smoke_file, "w") as h5:
                entry = h5.create_group("entry")
                one_d = entry.create_group("integrated_1d")
                one_d.create_dataset("frame_index", data=frames)
                q_dataset = one_d.create_dataset("q", data=q)
                q_dataset.attrs["units"] = "q_A^-1"
                one_d.create_dataset("intensity", data=stack)
                one_d.create_dataset("sigma", data=np.sqrt(stack))
                scan_data = entry.create_group("scan_data")
                scan_data.create_dataset("frame_index", data=frames)
                scan_data.create_dataset("elapsed_time", data=frames * 0.002)
                scan_data.create_dataset("i0", data=np.linspace(0.98, 1.02, len(frames)))
                cakes = entry.create_group("integrated_2d")
                cakes.create_dataset("frame_index", data=frames)
                cakes.create_dataset("q", data=q)
                cakes.create_dataset("chi", data=np.linspace(-30, 30, 12))
                cakes.create_dataset("intensity", data=np.broadcast_to(stack[:, None, :], (len(frames), 12, len(q))))
                frame_groups = entry.create_group("frames")
                for frame in frames:
                    frame_groups.create_group(f"frame_{frame:04d}").create_dataset("thumbnail", data=np.ones((8, 8), dtype=np.uint8))
            selected_paths = smoke_file
        else:
            assert processed_selection.exists(), f"Missing processed NeXus: {processed_selection}"
            selected_paths = processed_selection

        series = load_time_resolved_series(selected_paths, metadata_keys=("i0",), source_root=source_root)
        raw_status = "not requested"
        thumbnail = series.get_thumbnail(0)
        cake = series.get_cake(0)
        try:
            raw = series.get_raw(0)
            raw_status = f"raw shape={raw.shape}"
        except KeyError:
            raw_status = "raw source unavailable; thumbnail remains distinct"
        {"patterns": series.dataset.sizes["pattern"], "q_points": series.dataset.sizes["q"], "raw": raw_status, "cake_shape": cake.intensity.shape}
        """
    ),
    code(
        """
        normalized = normalize_monitor(series.dataset, "i0")
        normalized = normalize_reference_band(normalized, q_range=tuple(q_band.value), intensity_var="intensity_normalized", output_var="intensity_band_normalized")
        flagged = flag_normalization_outliers(normalized)
        binned = bin_time_resolved(flagged, bin_size=bin_size.value)
        zeroed = select_time_zero(binned, zero_pattern=0)
        waterfall = plot_time_resolved_waterfall(zeroed, intensity_var="intensity_band_normalized", log_intensity=True)
        pattern_viewer = PatternViewer(patterns=[(zeroed.q.values, zeroed.intensity_band_normalized.isel(pattern=0).values, "row 0")])
        image_viewer = ImageViewer(thumbnail, title="Stored thumbnail")
        display(widgets.HBox([pattern_viewer.widget, image_viewer.widget]))
        waterfall
        """
    ),
    code(
        """
        plan = PeakFitPlan(positions=(2.76,), model="gaussian", background="linear", sigma_init=0.02, center_bounds_delta=0.08)
        fits = fit_peak_series(zeroed, plan, intensity_var="intensity_band_normalized", q_range=(2.60, 2.95), pattern_indices=np.arange(min(8, zeroed.sizes["pattern"])))
        fits = flag_fit_quality(fits, max_center_error=0.02)
        lattice = add_lattice_results(fits, hkls=((1, 1, 1),))
        calibration = LinearThermalExpansion(float(lattice.lattice_mean_A.isel(fit_pattern=0)), 300.0, 9e-6)
        thermal = add_temperature_results(lattice, calibration, time_coord="time")
        if export_enabled.value:
            export_time_resolved_results(
                thermal,
                netcdf_path=export_directory / "time_resolved_results.nc",
                csv_path=export_directory / "time_resolved_scalars.csv",
            )
        display(plot_peak_fit_frame(thermal, 0))
        display(plot_thermal_history(thermal))
        {"fit_rows": thermal.sizes["fit_pattern"], "valid": int(thermal.fit_valid.sum()), "temperature_units": thermal.temperature_K.attrs["units"]}
        """
    ),
]


NB_RSM = [
    md(
        """
        # Reciprocal-Space Mapping

        The public RSM path separates a source, persisted geometry, and an
        `RSMPlan`. Smoke mode creates a bounded `RSMVolume` for slice review;
        real mode requires the matching `PixelQMap` and scan motor mapping.
        """
    ),
    code(
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display

        from xrd_tools.analysis import RSMPlan, run_rsm
        from xrd_tools.io import open_scan
        from xrd_tools.rsm import RSMVolume
        from xrd_tools.viz import plot_image
        """
    ),
    CONFIG,
    code(
        """
        processed_file = TEST_DATA / "processed.nxs"
        mapper = None  # Real mode: PixelQMap from the experiment's persisted geometry.
        diff_motors = ()  # Real mode: one persisted scan_data motor name per circle.
        slice_axis = widgets.Dropdown(options=("h", "k", "l"), value="l", description="integrate")
        compute = widgets.Button(description="Compute RSM", button_style="primary")
        status = widgets.HTML("<i>Compute is explicit; changing a slice does not regrid data.</i>")
        display(widgets.VBox([slice_axis, compute, status]))
        """
    ),
    code(
        """
        if SMOKE_MODE:
            h = np.linspace(-0.08, 0.08, 24)
            k = np.linspace(-0.06, 0.06, 20)
            l = np.linspace(0.90, 1.10, 18)
            hh, kk, ll = np.meshgrid(h, k, l, indexing="ij")
            volume = RSMVolume(h, k, l, np.exp(-0.5 * ((hh / 0.02) ** 2 + (kk / 0.018) ** 2 + ((ll - 1.0) / 0.03) ** 2)))
        else:
            assert processed_file.is_file(), f"Missing processed NeXus: {processed_file}"
            assert mapper is not None and diff_motors, "Set mapper and diff_motors from the experiment geometry."
            source = open_scan(processed_file)
            plan = RSMPlan(mapper=mapper, diff_motors=tuple(diff_motors), bins=(96, 96, 96))
            volume = run_rsm(plan, source).payload

        axis_a, axis_b, image, integrated = volume.get_slice(slice_axis.value)
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_image(ax, image.T, attrs={"xlabel": "axis 1", "ylabel": "axis 2", "title": f"RSM projection over {slice_axis.value}"}, cb_label="Intensity")
        plt.show()
        {"shape": volume.shape, "integrated_points": len(integrated), "bounds": volume.get_bounds()}
        """
    ),
]


NOTEBOOKS = {
    "01_batch_integration.ipynb": NB_BATCH_INTEGRATION,
    "02_multigeometry_stitching.ipynb": NB_STITCHING,
    "03_phase_and_peak_fitting.ipynb": NB_PHASE_PEAK,
    "04_batch_phase_fitting.ipynb": NB_BATCH_PHASE,
    "05_sin2psi_analysis.ipynb": NB_SIN2PSI,
    "06_headless_reduction_pipeline.ipynb": NB_REDUCTION,
    "07_reading_processed_nxs.ipynb": NB_READING,
    "08_time_resolved_xrd_analysis.ipynb": NB_TIME_RESOLVED,
    "09_reciprocal_space_mapping.ipynb": NB_RSM,
}


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        write(name, cells)
    print(f"Wrote {len(NOTEBOOKS)} clean notebooks to {ROOT}")
