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
        from IPython.display import clear_output, display

        from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
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
        npt_1d = widgets.BoundedIntText(value=256, min=32, max=4096, description="1-D points")
        npt_azim = widgets.BoundedIntText(value=72, min=8, max=720, description="chi points")
        run_button = widgets.Button(description="Run integration", button_style="primary")
        status = widgets.HTML("<i>Configure real paths, then run explicitly.</i>")
        output = widgets.Output()
        display(widgets.VBox([widgets.HBox([npt_1d, npt_azim]), run_button, status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"runs": 0, "result_1d": None, "result_2d": None}

        def run_integration(_=None):
            # Run both public integration operations with current widget values.
            with output:
                clear_output(wait=True)
                try:
                    if SMOKE_MODE:
                        q = np.linspace(1.0, 4.0, npt_1d.value)
                        chi = np.linspace(-35.0, 35.0, npt_azim.value)
                        intensity = 20 + 90 * np.exp(-0.5 * ((q - 2.35) / 0.07) ** 2)
                        cake = np.outer(intensity, 1.0 + 0.1 * np.cos(np.deg2rad(chi)))
                        result_1d = IntegrationResult1D(q, intensity, unit="q_A^-1")
                        result_2d = IntegrationResult2D(q, chi, cake, unit="q_A^-1", azimuthal_unit="chi_deg")
                    else:
                        assert image_file.is_file(), f"Missing detector image: {image_file}"
                        assert poni_file.is_file(), f"Missing PONI calibration: {poni_file}"
                        image, poni = read_image(image_file), load_poni(poni_file)
                        result_1d = integrate_1d(image, poni, npt=npt_1d.value)
                        result_2d = integrate_2d(image, poni, npt_rad=npt_1d.value, npt_azim=npt_azim.value)
                    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
                    plot_1d(axes[0], result_1d.radial, result_1d.intensity, fmt="-", attrs={"xlabel": f"q ({result_1d.unit})", "ylabel": "Intensity", "title": "1-D integration"})
                    axes[1].pcolormesh(result_2d.radial, result_2d.azimuthal, result_2d.intensity.T, shading="auto")
                    axes[1].set(xlabel=f"q ({result_2d.unit})", ylabel=result_2d.azimuthal_unit, title="2-D integration")
                    plt.show()
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, result_1d=result_1d, result_2d=result_2d)
                    status.value = f"<b>Integrated {npt_1d.value} q points and {npt_azim.value} chi points.</b>"
                except Exception as exc:
                    status.value = f"<b>Integration failed:</b> {exc}"
                    raise

        run_button.on_click(run_integration)
        NOTEBOOK_ACTIONS = {"run_integration": run_integration}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            run_integration()
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
        from IPython.display import clear_output, display

        from xrd_tools.analysis import StitchPlan, run_stitch
        from xrd_tools.core.containers import PONI
        from xrd_tools.core.scan import ScanFrame
        from xrd_tools.integrate import load_poni
        from xrd_tools.io import load_mask, read_image
        from xrd_tools.sources import MemoryFrameSource
        from xrd_tools.viz import plot_1d
        """
    ),
    CONFIG,
    code(
        """
        image_paths = []  # Real mode: naturally ordered detector images.
        poni_file = None  # Real mode: persisted base PONI.
        rotation_values = []  # Real mode: rot1 degrees, one per image.
        monitor_values = []   # Real mode: optional monitor, one per image.
        mask_file = None      # Real mode: optional detector mask.
        q_range = widgets.FloatRangeSlider(value=(1.0, 4.0), min=0.5, max=6.0, step=0.05, description="q range", continuous_update=False)
        monitor_key = widgets.Text(value="i0", description="monitor")
        use_mask = widgets.Checkbox(value=False, description="apply mask")
        compute = widgets.Button(description="Compute stitch", button_style="primary")
        status = widgets.HTML("<i>Compute performs both 1-D and 2-D StitchPlan runs.</i>")
        output = widgets.Output()
        display(widgets.VBox([q_range, monitor_key, use_mask, compute, status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"runs": 0, "stitch_1d": None, "stitch_2d": None}

        def _smoke_stitch_source():
            shape = (195, 487)
            poni = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2, poni2=shape[1] * 172e-6 / 2,
                        rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1e-10, detector="Pilatus100k")
            yy, xx = np.mgrid[:shape[0], :shape[1]]
            radius = np.sqrt((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2)
            base = 300 * np.exp(-((radius - 55) / 9) ** 2) + 2
            frames = [ScanFrame(index, image=base * (1 + 0.03 * index), metadata={"rot1": 4.0 * index, "i0": 1.0 + 0.02 * index}) for index in range(3)]
            return MemoryFrameSource(frames, name="synthetic_stitch"), poni

        def _real_stitch_source():
            assert image_paths, "Configure image_paths for real stitching"
            assert poni_file is not None, "Configure a persisted base poni_file"
            paths = sorted(map(Path, image_paths), key=lambda path: path.name)
            assert len(rotation_values) == len(paths), "rotation_values must match image_paths"
            if monitor_values:
                assert len(monitor_values) == len(paths), "monitor_values must match image_paths"
            mask = load_mask(mask_file) if use_mask.value and mask_file else None
            frames = [ScanFrame(index, image=read_image(path, mask=mask), metadata={"rot1": float(rotation_values[index]), "i0": float(monitor_values[index]) if monitor_values else 1.0}) for index, path in enumerate(paths)]
            return MemoryFrameSource(frames, name="real_stitch"), load_poni(poni_file)

        def compute_stitch(_=None):
            with output:
                clear_output(wait=True)
                try:
                    source, poni = _smoke_stitch_source() if SMOKE_MODE else _real_stitch_source()
                    common = dict(base_poni=poni, rot1_key="rot1", monitor_key=monitor_key.value or None,
                                  radial_range=tuple(q_range.value), npt_1d=180, npt_rad_2d=120, npt_azim_2d=48)
                    one_d = run_stitch(StitchPlan(mode="1d", **common), source).payload
                    two_d = run_stitch(StitchPlan(mode="2d", **common), source).payload
                    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
                    plot_1d(axes[0], one_d.radial, one_d.intensity, fmt="-", attrs={"xlabel": one_d.unit, "ylabel": "Intensity", "title": "StitchPlan 1-D"})
                    axes[1].pcolormesh(two_d.radial, two_d.azimuthal, two_d.intensity.T, shading="auto")
                    axes[1].set(xlabel=two_d.unit, ylabel=two_d.azimuthal_unit, title="StitchPlan 2-D")
                    plt.show()
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, stitch_1d=one_d, stitch_2d=two_d)
                    status.value = f"<b>Stitched {len(source.frame_indices)} images in 1-D and 2-D.</b>"
                except Exception as exc:
                    status.value = f"<b>Stitch failed:</b> {exc}"
                    raise

        compute.on_click(compute_stitch)
        NOTEBOOK_ACTIONS = {"compute_stitch": compute_stitch}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            compute_stitch()
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
        from IPython.display import clear_output, display

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
        output = widgets.Output()
        status = widgets.HTML("<i>Change fit settings, then use the shared Fit action.</i>")
        NOTEBOOK_STATE = {"runs": 0, "outcome": None}

        def run_pilot(params=None):
            params = params or controls.get_params()
            with output:
                clear_output(wait=True)
                try:
                    positions = tuple(params["positions"] or (2.76,))
                    plan = PeakFitPlan(positions=positions, model=params["model"], background=params["background"] if params["background"] in {"none", "constant", "linear"} else "linear", sigma_init=params["sigma_init"], sigma_bounds=params["sigma_bounds"], center_bounds_delta=params["center_bounds_delta"], fit_kwargs={"method": "leastsq"})
                    outcome = PeakFitAnalyzer(plan).analyze(AnalysisInput(label="pilot", x=q, y=intensity, x_unit="q_A^-1"))
                    assert outcome.ok, outcome.message
                    display(plot_peak_fit(q, intensity, outcome.result.payload, title="Pilot peak fit"))
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, outcome=outcome, plan=plan)
                    status.value = "<b>Pilot fit complete.</b>"
                except Exception as exc:
                    status.value = f"<b>Pilot fit failed:</b> {exc}"
                    raise

        controls = PeakFitControls(on_fit=run_pilot)
        controls.n_peaks.value = 1
        controls.peak_positions.value = "2.76"
        controls.peak_model.value = "gaussian"
        controls.bg_model.value = "linear"
        controls.sigma_init.value = 0.02
        controls.q_min.value, controls.q_max.value = 2.3, 3.2
        viewer = PatternViewer(patterns=[(q, intensity, "pilot")])
        display(widgets.VBox([controls.widget, viewer.widget, status, output]))
        NOTEBOOK_ACTIONS = {"run_pilot": run_pilot}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            run_pilot()
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
        from IPython.display import clear_output, display

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
        output = widgets.Output()
        display(widgets.VBox([quality, widgets.HBox([run_button]), status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"runs": 0, "outcomes": []}

        def run_batch_fit(_=None):
            with output:
                clear_output(wait=True)
                try:
                    patterns = [8 + 80 * np.exp(-0.5 * ((q - center) / quality.value) ** 2) for center in (2.755, 2.760, 2.765)]
                    analyzer = PeakFitAnalyzer(PeakFitPlan(positions=(2.76,), model="gaussian", background="linear", sigma_init=quality.value))
                    outcomes = run_batch(analyzer, [AnalysisInput(str(i), q, y, x_unit="q_A^-1") for i, y in enumerate(patterns)])
                    labels, columns = batch_params_table(outcomes)
                    assert len(labels) == len(patterns) == len(columns["center_0"])
                    display(columns)
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, outcomes=outcomes, columns=columns)
                    status.value = f"<b>Fit {len(outcomes)} aligned patterns with width={quality.value:.3f}.</b>"
                except Exception as exc:
                    status.value = f"<b>Batch fit failed:</b> {exc}"
                    raise

        run_button.on_click(run_batch_fit)
        NOTEBOOK_ACTIONS = {"run_batch_fit": run_batch_fit}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            run_batch_fit()
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
        from IPython.display import clear_output, display

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
        status = widgets.HTML("<i>Fit sectors explicitly; q changes do not rerun the fit.</i>")
        output = widgets.Output()
        display(widgets.VBox([q_range, fit_button, status, output]))
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
        NOTEBOOK_STATE = {"runs": 0, "result": None}

        def fit_sectors(_=None):
            with output:
                clear_output(wait=True)
                try:
                    result = run_sin2psi(Sin2PsiPlan(q_range=tuple(q_range.value), chi_width=7.0), polar).payload
                    assert np.isfinite(result.d0)
                    display({"d0_A": result.d0, "slope_A": result.slope, "r_squared": result.r_squared})
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, result=result)
                    status.value = f"<b>Fitted sectors over q={tuple(q_range.value)}.</b>"
                except Exception as exc:
                    status.value = f"<b>Sector fit failed:</b> {exc}"
                    raise

        fit_button.on_click(fit_sectors)
        NOTEBOOK_ACTIONS = {"fit_sectors": fit_sectors}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            fit_sectors()
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
        from IPython.display import clear_output, display

        from xrd_tools.core.containers import PONI
        from xrd_tools.core.scan import Scan, ScanFrame
        from xrd_tools.integrate import load_poni
        from xrd_tools.io import read_image
        from xrd_tools.reduction import CancelToken, Integration1DPlan, Integration2DPlan, MemorySink, ReductionPlan, run_reduction
        """
    ),
    CONFIG,
    code(
        """
        image_paths = []  # Real mode: bounded ordered detector images.
        poni_file = None  # Real mode: calibration for image_paths.
        npt_1d = widgets.BoundedIntText(value=128, min=32, max=4096, description="1-D points")
        include_2d = widgets.Checkbox(value=True, description="also integrate 2-D")
        run_button = widgets.Button(description="Run reduction", button_style="primary")
        cancel_button = widgets.Button(description="Cancel active run")
        chunk_size = widgets.BoundedIntText(value=4, min=1, max=64, description="chunk")
        progress = widgets.IntProgress(value=0, min=0, max=1, description="frames")
        status = widgets.HTML("<i>Run creates a real MemorySink; cancel requests the next frame boundary.</i>")
        output = widgets.Output()
        display(widgets.VBox([widgets.HBox([npt_1d, chunk_size, include_2d]), widgets.HBox([run_button, cancel_button]), progress, status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"runs": 0, "sink": None, "result": None, "cancel_token": None}

        def _smoke_scan():
            shape = (195, 487)
            poni = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2, poni2=shape[1] * 172e-6 / 2,
                        rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1e-10, detector="Pilatus100k")
            yy, xx = np.mgrid[:shape[0], :shape[1]]
            radius = np.sqrt((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2)
            image = 500 * np.exp(-((radius - 60) / 10) ** 2) + 3
            return Scan("synthetic_reduction", [ScanFrame(index, image=image * (1 + 0.05 * index)) for index in range(2)], poni=poni)

        def _real_scan():
            assert image_paths and poni_file is not None, "Configure image_paths and poni_file for real reduction"
            paths = sorted(map(Path, image_paths), key=lambda path: path.name)
            return Scan("configured_reduction", [ScanFrame(index, image=read_image(path), source_path=path) for index, path in enumerate(paths)], poni=load_poni(poni_file))

        def cancel_active_run(_=None):
            token = NOTEBOOK_STATE.get("cancel_token")
            if token is not None:
                token.cancel()
                status.value = "<b>Cancellation requested at the next frame boundary.</b>"

        def run_pipeline(_=None):
            with output:
                clear_output(wait=True)
                try:
                    scan = _smoke_scan() if SMOKE_MODE else _real_scan()
                    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=npt_1d.value), integration_2d=Integration2DPlan(npt_rad=max(32, npt_1d.value // 2), npt_azim=32) if include_2d.value else None)
                    sink, token = MemorySink(), CancelToken()
                    NOTEBOOK_STATE["cancel_token"] = token
                    progress.max, progress.value = len(scan), 0
                    def update(event):
                        progress.value = min(event.completed, progress.max)
                    result = run_reduction(plan, scan, sink=sink, chunk_size=chunk_size.value, progress_cb=update, cancel_token=token)
                    assert sink.frames and result.n_processed == len(sink.frames)
                    display({"processed": result.n_processed, "cancelled": result.cancelled, "sink_frames": sorted(sink.frames)})
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, sink=sink, result=result)
                    status.value = f"<b>Reduced {result.n_processed} frames into MemorySink.</b>"
                except Exception as exc:
                    status.value = f"<b>Reduction failed:</b> {exc}"
                    raise
                finally:
                    NOTEBOOK_STATE["cancel_token"] = None

        run_button.on_click(run_pipeline)
        cancel_button.on_click(cancel_active_run)
        NOTEBOOK_ACTIONS = {"run_reduction": run_pipeline, "cancel_reduction": cancel_active_run}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            run_pipeline()
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
        import h5py
        import tempfile
        from IPython.display import clear_output, display

        from xrd_tools.gui.widgets import ImageViewer, PatternViewer
        from xrd_tools.io import open_scan
        """
    ),
    CONFIG,
    code(
        """
        processed_file = TEST_DATA / "xdart_processed_data" / "Pt_10nm_00013.nxs"
        frame_selector = widgets.BoundedIntText(value=0, min=0, max=999999, description="frame label")
        source_root = widgets.Text(value="", description="source root")
        inspect_button = widgets.Button(description="Inspect frame", button_style="primary")
        status = widgets.HTML("<i>Inspect reads only the selected 1-D/cake/raw-or-thumbnail row.</i>")
        output = widgets.Output()
        display(widgets.VBox([source_root, frame_selector, inspect_button, status, output]))
        """
    ),
    code(
        """
        if SMOKE_MODE:
            smoke_file = Path(tempfile.gettempdir()) / "xdart_notebook_processed_read_smoke.nxs"
            q = np.linspace(1.0, 4.0, 180)
            frames = np.array([3, 7], dtype=np.int64)
            stack = np.vstack([12 + np.sin(q) ** 2, 13 + np.cos(q) ** 2]).astype("float32")
            with h5py.File(smoke_file, "w") as h5:
                entry = h5.create_group("entry")
                one_d = entry.create_group("integrated_1d")
                one_d.create_dataset("frame_index", data=frames)
                q_data = one_d.create_dataset("q", data=q); q_data.attrs["units"] = "q_A^-1"
                one_d.create_dataset("intensity", data=stack)
                two_d = entry.create_group("integrated_2d")
                two_d.create_dataset("frame_index", data=frames)
                two_d.create_dataset("q", data=q)
                two_d.create_dataset("chi", data=np.linspace(-20, 20, 12))
                two_d.create_dataset("intensity", data=np.broadcast_to(stack[:, None, :], (2, 12, q.size)))
                groups = entry.create_group("frames")
                for frame in frames:
                    groups.create_group(f"frame_{frame:04d}").create_dataset("thumbnail", data=np.arange(64, dtype="uint8").reshape(8, 8))
            processed_file = smoke_file

        NOTEBOOK_STATE = {"runs": 0, "scan": None, "selected": None, "raw_available": False}

        def inspect_frame(_=None):
            with output:
                clear_output(wait=True)
                try:
                    scan = open_scan(processed_file, source_root=source_root.value or None)
                    labels = scan.frame_indices
                    frame_selector.min, frame_selector.max = min(labels), max(labels)
                    selected = int(frame_selector.value)
                    if selected not in labels:
                        selected = labels[0]
                        frame_selector.value = selected
                    pattern, cake = scan.get_1d(selected), scan.get_2d(selected)
                    thumbnail = scan.get_thumbnail(selected)
                    try:
                        image, image_title = scan.load_frame(selected), "Full raw frame"
                        raw_available = True
                    except (KeyError, FileNotFoundError, ValueError):
                        image, image_title, raw_available = thumbnail, "Stored thumbnail (raw unavailable)", False
                    pattern_viewer = PatternViewer(patterns=[(pattern.q, pattern.intensity, f"frame label {selected}")])
                    image_viewer = ImageViewer(image, title=image_title)
                    tabs = widgets.Tab(children=[pattern_viewer.widget, image_viewer.widget])
                    tabs.set_title(0, "1-D pattern")
                    tabs.set_title(1, "raw / thumbnail")
                    display(tabs)
                    display({"scan": scan.path.name, "frame_label": selected, "available_labels": labels, "cake_shape": cake.intensity.shape, "raw_available": raw_available})
                    NOTEBOOK_STATE.update(runs=NOTEBOOK_STATE["runs"] + 1, scan=scan, selected=selected, raw_available=raw_available, cake=cake)
                    status.value = f"<b>Inspected scan-qualified frame {scan.path.name}:{selected}.</b>"
                except Exception as exc:
                    status.value = f"<b>Frame inspection failed:</b> {exc}"
                    raise

        inspect_button.on_click(inspect_frame)
        frame_selector.observe(lambda change: inspect_frame() if NOTEBOOK_STATE["scan"] is not None else None, names="value")
        NOTEBOOK_ACTIONS = {"inspect_frame": inspect_frame}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            inspect_frame()
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
        from IPython.display import clear_output, display

        from xrd_tools.analysis import (
            LinearThermalExpansion, PeakFitPlan, add_lattice_results,
            add_temperature_results, bin_time_resolved, fit_peak_series,
            export_time_resolved_results,
            discover_processed_scans, flag_fit_quality, flag_normalization_outliers, load_time_resolved_series,
            normalize_monitor, normalize_reference_band, select_time_zero,
        )
        from xrd_tools.gui.widgets import ImageViewer, PatternViewer, PeakFitControls
        from xrd_tools.viz import plot_peak_fit_frame, plot_thermal_history, plot_time_resolved_waterfall
        """
    ),
    CONFIG,
    code(
        """
        processed_root = TEST_DATA / "xdart_processed_data"
        processed_file = widgets.Text(value=str(processed_root / "Pt_test_burst_00007.nxs"), description="processed file")
        processed_folder = widgets.Text(value=str(processed_root), description="processed folder")
        selection_mode = widgets.ToggleButtons(options=("file", "folder"), value="file", description="load")
        scan_filter = widgets.Text(value="*.nxs", description="scan filter")
        discover_button = widgets.Button(description="Find processed scans")
        scan_selection = widgets.SelectMultiple(options=(), description="selected scans", layout=widgets.Layout(width="620px", height="120px"))
        raw_root = widgets.Text(value=str(TEST_DATA) if TEST_DATA.exists() else "", description="raw root")
        time_key = widgets.Text(value="", description="time key")
        time_unit = widgets.Dropdown(options=("s", "ms", "us"), value="s", description="time unit")
        frame_period_ms = widgets.BoundedFloatText(value=2.0, min=0.001, max=1e6, step=0.1, description="period ms")
        monitor_method = widgets.Dropdown(options=("reference band", "monitor"), value="reference band", description="normalize")
        monitor_key = widgets.Text(value="i0", description="monitor key")
        q_band = widgets.FloatRangeSlider(value=(3.05, 3.20), min=2.4, max=3.3, step=0.01, description="reference q", continuous_update=False)
        bin_size = widgets.BoundedIntText(value=2, min=1, max=100, description="bin size")
        frame_row = widgets.BoundedIntText(value=0, min=0, max=0, description="row")
        batch_limit = widgets.BoundedIntText(value=8, min=1, max=1000, description="batch rows")
        thermal_time_basis = widgets.Dropdown(options=("automatic", "time", "sequence_time"), value="automatic", description="thermal time")
        load_button = widgets.Button(description="Load processed", button_style="primary")
        preprocess_button = widgets.Button(description="Apply preprocessing")
        inspect_button = widgets.Button(description="Inspect lazy row")
        pilot_button = widgets.Button(description="Run pilot fit", button_style="primary")
        batch_button = widgets.Button(description="Run bounded batch", button_style="primary")
        export_button = widgets.Button(description="Export compact results")
        export_directory = Path(os.environ.get("XDART_NOTEBOOK_OUTPUT", tempfile.gettempdir()))
        status = widgets.HTML("<i>Load, preprocess, and fit actions are explicit; row selection only reads cached/lazy data.</i>")
        output = widgets.Output()
        display(widgets.VBox([selection_mode, processed_file, processed_folder, widgets.HBox([scan_filter, discover_button]), scan_selection, raw_root, widgets.HBox([time_key, time_unit, frame_period_ms]), widgets.HBox([monitor_method, monitor_key]), q_band, widgets.HBox([bin_size, frame_row, batch_limit, thermal_time_basis]), widgets.HBox([load_button, preprocess_button, inspect_button]), widgets.HBox([pilot_button, batch_button, export_button]), status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"loads": 0, "preprocesses": 0, "series": None, "prepared": None, "fits": None, "thermal": None, "raw_reads": 0, "discovered_scans": (), "selected_scans": (), "thermal_time_coord": None}

        def _smoke_file():
            path = Path(tempfile.gettempdir()) / "xdart_notebook_time_resolved_smoke.nxs"
            q, frames = np.linspace(2.45, 3.30, 240), np.arange(16, dtype=np.int64)
            centers = 2.765 - 0.0008 * frames
            stack = np.array([15 + 180 * np.exp(-0.5 * ((q - center) / 0.018) ** 2) for center in centers], dtype=np.float32)
            with h5py.File(path, "w") as h5:
                entry = h5.create_group("entry")
                one_d = entry.create_group("integrated_1d"); one_d.create_dataset("frame_index", data=frames)
                q_data = one_d.create_dataset("q", data=q); q_data.attrs["units"] = "q_A^-1"
                one_d.create_dataset("intensity", data=stack); one_d.create_dataset("sigma", data=np.sqrt(stack))
                scan_data = entry.create_group("scan_data"); scan_data.create_dataset("frame_index", data=frames)
                scan_data.create_dataset("elapsed", data=frames * 2.0); scan_data.create_dataset("i0", data=np.linspace(0.98, 1.02, len(frames)))
                cakes = entry.create_group("integrated_2d"); cakes.create_dataset("frame_index", data=frames); cakes.create_dataset("q", data=q); cakes.create_dataset("chi", data=np.linspace(-30, 30, 12)); cakes.create_dataset("intensity", data=np.broadcast_to(stack[:, None, :], (len(frames), 12, len(q))))
                groups = entry.create_group("frames")
                for frame in frames:
                    groups.create_group(f"frame_{frame:04d}").create_dataset("thumbnail", data=np.ones((8, 8), dtype=np.uint8))
            return path

        def discover_folder_scans(_=None):
            with output:
                clear_output(wait=True)
                try:
                    assert selection_mode.value == "folder", "Choose folder mode before discovering scans"
                    root = Path(processed_folder.value).expanduser()
                    pattern = scan_filter.value.strip() or "*.nxs"
                    paths = discover_processed_scans(root, pattern=pattern)
                    assert paths, f"No processed 1-D scans match {pattern!r} under {root}"
                    scan_selection.options = [(path.name, str(path)) for path in paths]
                    scan_selection.value = ()
                    NOTEBOOK_STATE["discovered_scans"] = tuple(paths)
                    display({"discovered": [path.name for path in paths], "next": "select one or more scans, then load"})
                    status.value = f"<b>Discovered {len(paths)} naturally sorted processed scans; select the subset to load.</b>"
                except Exception as exc:
                    status.value = f"<b>Scan discovery failed:</b> {exc}"
                    raise

        def _selected_paths():
            if SMOKE_MODE:
                return _smoke_file()
            if selection_mode.value == "file":
                candidate = Path(processed_file.value).expanduser()
                assert candidate.is_file(), f"Missing processed selection: {candidate}"
                return candidate
            paths = tuple(Path(value) for value in scan_selection.value)
            assert paths, "Discover a folder and select one or more processed scans before loading"
            discovered = set(NOTEBOOK_STATE["discovered_scans"])
            assert set(paths).issubset(discovered), "Selected scans are not from the current discovered folder"
            return paths

        def load_processed(_=None):
            with output:
                clear_output(wait=True)
                try:
                    selected_key = "elapsed" if SMOKE_MODE else time_key.value.strip() or None
                    selected_unit = "ms" if SMOKE_MODE else (time_unit.value if selected_key else None)
                    paths = _selected_paths()
                    selected_names = [paths.name] if isinstance(paths, Path) else [path.name for path in paths]
                    series = load_time_resolved_series(paths, frame_period_s=frame_period_ms.value / 1000.0, time_key=selected_key, time_unit=selected_unit, metadata_keys=(monitor_key.value.strip(),) if monitor_key.value.strip() else (), source_root=raw_root.value.strip() or None)
                    frame_row.max, frame_row.value = series.dataset.sizes["pattern"] - 1, 0
                    NOTEBOOK_STATE.update(loads=NOTEBOOK_STATE["loads"] + 1, series=series, prepared=None, fits=None, thermal=None, selected_scans=tuple(selected_names), thermal_time_coord=None)
                    status.value = f"<b>Loaded {series.dataset.sizes['pattern']} scan-qualified patterns from {len(selected_names)} selected scan(s).</b>"
                    display({"selected_scans": selected_names, "patterns": series.dataset.sizes["pattern"], "q_points": series.dataset.sizes["q"], "time_units": series.dataset.coords["time"].attrs["units"]})
                except Exception as exc:
                    status.value = f"<b>Load failed:</b> {exc}"
                    raise

        def apply_preprocessing(_=None):
            with output:
                clear_output(wait=True)
                try:
                    series = NOTEBOOK_STATE["series"]
                    assert series is not None, "Load processed data first"
                    dataset, fallback = series.dataset, False
                    if monitor_method.value == "monitor":
                        try:
                            candidate = normalize_monitor(dataset, monitor_key.value.strip())
                            if bool(candidate["monitor_normalization_valid"].any()):
                                dataset, source_var = candidate, "intensity_normalized"
                            else:
                                fallback, source_var = True, "intensity"
                        except (KeyError, ValueError):
                            fallback, source_var = True, "intensity"
                    else:
                        source_var = "intensity"
                    normalized = normalize_reference_band(dataset, q_range=tuple(q_band.value), intensity_var=source_var, output_var="intensity_band_normalized")
                    prepared = select_time_zero(bin_time_resolved(flag_normalization_outliers(normalized), bin_size=bin_size.value), zero_pattern=0)
                    NOTEBOOK_STATE.update(preprocesses=NOTEBOOK_STATE["preprocesses"] + 1, prepared=prepared, fits=None, thermal=None)
                    display(plot_time_resolved_waterfall(prepared, intensity_var="intensity_band_normalized", log_intensity=True))
                    status.value = "<b>Reference-band preprocessing complete.</b>" if not fallback else "<b>Selected monitor was unavailable; used documented reference-band fallback.</b>"
                except Exception as exc:
                    status.value = f"<b>Preprocessing failed:</b> {exc}"
                    raise

        def inspect_lazy_row(_=None):
            with output:
                clear_output(wait=True)
                try:
                    series = NOTEBOOK_STATE["series"]
                    assert series is not None, "Load processed data first"
                    row = int(frame_row.value)
                    thumbnail, cake = series.get_thumbnail(row), series.get_cake(row)
                    try:
                        image, title, raw_available = series.get_raw(row), "Full raw detector frame", True
                    except (KeyError, FileNotFoundError, ValueError):
                        image, title, raw_available = thumbnail, "Stored thumbnail (raw unavailable)", False
                    pattern = series.dataset.isel(pattern=row)
                    display(widgets.HBox([PatternViewer(patterns=[(pattern.q.values, pattern.intensity.values, f"row {row}")]).widget, ImageViewer(image, title=title).widget]))
                    display({"scan": str(pattern.scan_name.item()), "frame_label": int(pattern.frame_label.item()), "cake_shape": cake.intensity.shape, "raw_available": raw_available})
                    NOTEBOOK_STATE["raw_reads"] += 1
                except Exception as exc:
                    status.value = f"<b>Lazy inspection failed:</b> {exc}"
                    raise

        def _plan_from_controls(params):
            positions = tuple(params["positions"] or (2.76,))
            return PeakFitPlan(positions=positions, model=params["model"], background=params["background"] if params["background"] in {"none", "constant", "linear"} else "linear", sigma_init=params["sigma_init"], sigma_bounds=params["sigma_bounds"], center_bounds_delta=params["center_bounds_delta"], fit_kwargs={"method": "leastsq"})

        def run_pilot(params=None):
            with output:
                clear_output(wait=True)
                try:
                    prepared = NOTEBOOK_STATE["prepared"]
                    assert prepared is not None, "Apply preprocessing first"
                    fits = fit_peak_series(prepared, _plan_from_controls(params or fit_controls.get_params()), intensity_var="intensity_band_normalized", q_range=(fit_controls.q_min.value, fit_controls.q_max.value), pattern_indices=[min(int(frame_row.value), prepared.sizes["pattern"] - 1)])
                    NOTEBOOK_STATE["fits"] = flag_fit_quality(fits, max_center_error=0.02)
                    display(plot_peak_fit_frame(NOTEBOOK_STATE["fits"], 0))
                    status.value = "<b>Pilot fit complete.</b>"
                except Exception as exc:
                    status.value = f"<b>Pilot fit failed:</b> {exc}"
                    raise

        fit_controls = PeakFitControls(on_fit=run_pilot)
        peak_center = 2.76 if SMOKE_MODE else 1.56
        fit_controls.n_peaks.value = 1; fit_controls.peak_positions.value = str(peak_center); fit_controls.peak_model.value = "gaussian"; fit_controls.bg_model.value = "linear"; fit_controls.sigma_init.value = 0.02; fit_controls.q_min.value, fit_controls.q_max.value = peak_center - 0.16, peak_center + 0.16
        display(fit_controls.widget)

        def _thermal_time_coordinate(dataset):
            scan_count = len(np.unique(dataset.coords["scan_index"].values))
            requested = thermal_time_basis.value
            coordinate = requested if requested != "automatic" else ("time" if scan_count == 1 else "sequence_time")
            values = np.asarray(dataset.coords[coordinate].values, dtype=float)
            units = str(dataset.coords[coordinate].attrs.get("units", ""))
            if units not in {"s", "second", "seconds"} or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(
                    f"{coordinate!r} is not a finite strictly increasing physical-seconds coordinate; "
                    "select one scan or provide proven cross-scan cadence before deriving K/s"
                )
            return coordinate

        def run_bounded_batch(_=None):
            with output:
                clear_output(wait=True)
                try:
                    prepared = NOTEBOOK_STATE["prepared"]
                    assert prepared is not None, "Apply preprocessing first"
                    fits = flag_fit_quality(fit_peak_series(prepared, _plan_from_controls(fit_controls.get_params()), intensity_var="intensity_band_normalized", q_range=(fit_controls.q_min.value, fit_controls.q_max.value), pattern_indices=np.arange(min(batch_limit.value, prepared.sizes["pattern"]))), max_center_error=0.02)
                    lattice = add_lattice_results(fits, hkls=((1, 1, 1),))
                    calibration = LinearThermalExpansion(float(lattice.lattice_mean_A.isel(fit_pattern=0)), 300.0, 9e-6)
                    time_coordinate = _thermal_time_coordinate(lattice)
                    thermal = add_temperature_results(lattice, calibration, time_coord=time_coordinate)
                    NOTEBOOK_STATE.update(fits=fits, thermal=thermal, thermal_time_coord=time_coordinate)
                    display(plot_thermal_history(thermal))
                    status.value = f"<b>Saved {thermal.sizes['fit_pattern']} bounded fit rows using {time_coordinate} for K/s.</b>"
                except Exception as exc:
                    status.value = f"<b>Batch fit failed:</b> {exc}"
                    raise

        def export_compact(_=None):
            thermal = NOTEBOOK_STATE["thermal"]
            assert thermal is not None, "Run bounded batch before exporting"
            paths = export_time_resolved_results(thermal, netcdf_path=export_directory / "time_resolved_results.nc", csv_path=export_directory / "time_resolved_scalars.csv")
            status.value = f"<b>Exported compact results to {paths['netcdf'].parent}.</b>"
            return paths

        discover_button.on_click(discover_folder_scans); load_button.on_click(load_processed); preprocess_button.on_click(apply_preprocessing); inspect_button.on_click(inspect_lazy_row); pilot_button.on_click(run_pilot); batch_button.on_click(run_bounded_batch); export_button.on_click(export_compact)
        frame_row.observe(lambda change: inspect_lazy_row() if NOTEBOOK_STATE["series"] is not None else None, names="value")
        NOTEBOOK_ACTIONS = {"discover_folder_scans": discover_folder_scans, "load_processed": load_processed, "apply_preprocessing": apply_preprocessing, "inspect_lazy_row": inspect_lazy_row, "run_pilot": run_pilot, "run_bounded_batch": run_bounded_batch, "export_compact": export_compact}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            load_processed(); apply_preprocessing(); inspect_lazy_row(); run_pilot(); run_bounded_batch()
        """
    ),
]


NB_RSM = [
    md(
        """
        # Reciprocal-Space Mapping

        The public RSM path separates a source, persisted geometry, and an
        `RSMPlan`. Smoke mode creates a bounded `RSMVolume` for slice review;
        real mode requires the matching `PixelQMap`, authenticated physical UB,
        and scan motor mapping. HKL is the established/default frame; Cartesian
        Q must be selected explicitly.
        """
    ),
    code(
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        from xrd_tools.analysis import RSMPlan, run_rsm
        from xrd_tools.io import open_scan
        from xrd_tools.rsm import RSMCoordinateFrame, RSMVolume
        from xrd_tools.viz import plot_image
        """
    ),
    CONFIG,
    code(
        """
        processed_file = TEST_DATA / "processed.nexus"
        mapper = None  # Real mode: PixelQMap from the experiment's persisted geometry.
        UB = None  # Real mode: authenticated physical 3x3 UB for this source.
        diff_motors = ()  # Real mode: one persisted scan_data motor name per circle.
        slice_axis = widgets.Dropdown(options=("h", "k", "l"), value="l", description="integrate")
        compute = widgets.Button(description="Compute RSM", button_style="primary")
        status = widgets.HTML("<i>Compute is explicit; changing a slice does not regrid data.</i>")
        output = widgets.Output()
        display(widgets.VBox([slice_axis, compute, status, output]))
        """
    ),
    code(
        """
        NOTEBOOK_STATE = {"computes": 0, "slice_draws": 0, "volume": None}

        def _synthetic_volume():
            h, k, l = np.linspace(-0.08, 0.08, 24), np.linspace(-0.06, 0.06, 20), np.linspace(0.90, 1.10, 18)
            hh, kk, ll = np.meshgrid(h, k, l, indexing="ij")
            return RSMVolume(h, k, l, np.exp(-0.5 * ((hh / 0.02) ** 2 + (kk / 0.018) ** 2 + ((ll - 1.0) / 0.03) ** 2)))

        def draw_cached_slice(_=None):
            volume = NOTEBOOK_STATE["volume"]
            if volume is None:
                return
            with output:
                clear_output(wait=True)
                axis_a, axis_b, image, integrated = volume.get_slice(slice_axis.value)
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_image(ax, image.T, attrs={"xlabel": "axis 1", "ylabel": "axis 2", "title": f"RSM projection over {slice_axis.value}"}, cb_label="Intensity")
                plt.show()
                display({"shape": volume.shape, "integrated_points": len(integrated), "bounds": volume.get_bounds()})
                NOTEBOOK_STATE["slice_draws"] += 1

        def compute_rsm(_=None):
            try:
                if SMOKE_MODE:
                    volume = _synthetic_volume()
                else:
                    assert processed_file.is_file(), f"Missing processed NeXus: {processed_file}"
                    assert mapper is not None and diff_motors, "Set mapper and diff_motors from the experiment geometry."
                    assert UB is not None, "Set the authenticated physical UB for this source."
                    ub = np.asarray(UB, dtype=np.float64)
                    assert ub.shape == (3, 3) and np.all(np.isfinite(ub)), "UB must be a finite 3x3 matrix."
                    volume = run_rsm(
                        RSMPlan(
                            mapper=mapper,
                            diff_motors=tuple(diff_motors),
                            bins=(96, 96, 96),
                            UB=ub,
                            coordinate_frame=RSMCoordinateFrame.HKL,
                        ),
                        open_scan(processed_file),
                    ).payload
                NOTEBOOK_STATE.update(computes=NOTEBOOK_STATE["computes"] + 1, volume=volume)
                status.value = "<b>RSM gridding complete; slice changes redraw cached volume only.</b>"
                draw_cached_slice()
            except Exception as exc:
                status.value = f"<b>RSM compute failed:</b> {exc}"
                raise

        compute.on_click(compute_rsm)
        slice_axis.observe(draw_cached_slice, names="value")
        NOTEBOOK_ACTIONS = {"compute_rsm": compute_rsm, "draw_cached_slice": draw_cached_slice}
        if SMOKE_MODE or os.environ.get("XDART_NOTEBOOK_AUTORUN") == "1":
            compute_rsm()
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
