"""NEXT-128-PEAK-FIT T7: exported notebooks replay a file-sourced background template (headless).

Frozen before the product change (DESIGN-PEAK-FIT-128.md, Codex REVIEW-1 R1 and T7, and the PLAN's contract
revisions). Both tools, both file-source kinds and both rendered formats reproduce the fit the GUI route makes (the
public analyzer on the same plan settings) through the common headless route. The notebooks run from another
directory with explicit roots: the sidecars under BUNDLE_ROOT, the template (and the CIF) under DATA_ROOT, both
named relative. Interactive Apply and a compact batch whose inputs carry different template scales show the template
was not dropped. Uses the existing rendered-science executor pattern (test_fit_notebook_export.py).
"""

from __future__ import annotations

import importlib.util
import json
from importlib import import_module
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import write_frame_records
from tests.core.v2_fixture_factory import current_entry

X = np.linspace(1.0, 3.0, 401)
SCALES = (2.0, 3.0)
CONTROLS = ("input_selector", "overlay_selector", "comparison_selector", "log_display", "fit_range", "region_ranges",
            "phase_selector", "settings_editor", "axis_unit_selector", "display_min", "display_max", "slice_axis",
            "fit_apply", "fit_batch")


def _owner():
    return import_module("xrd_tools.analysis.background_template")


def _substrate(x):
    return 50.0 + 10.0 * np.sin(3.0 * x)


def _write_template(data, kind, x):
    """The substrate under DATA_ROOT, named relative: an .xye stored in nm^-1 (quoted, non-ASCII name) or one NeXus
    single-result frame in A^-1. Returns the relative source and the absolute one the GUI route holds."""
    owner = _owner()
    if kind == "xye":
        name = "iq_sub 'α'.xye"
        np.savetxt(data / name, np.column_stack([x * 10.0, _substrate(x)]))
        return (owner.BackgroundTemplateSource("xye", name, unit="q_nm^-1"),
                owner.BackgroundTemplateSource("xye", str(data / name), unit="q_nm^-1"))
    view = FrameView(label=3, axis_1d=axis_from_unit("q_A^-1", x), intensity_1d=_substrate(x))
    with h5py.File(data / "substrate.nexus", "w") as handle:
        write_frame_records(current_entry(handle), [FrameRecord.from_view(view)])
    return (owner.BackgroundTemplateSource("nexus", "substrate.nexus", frame=3, mode="default"),
            owner.BackgroundTemplateSource("nexus", str(data / "substrate.nexus"), frame=3, mode="default"))


def _peak_snapshot(index):
    y = SCALES[index] * _substrate(X) + 120.0 * np.exp(-0.5 * ((X - 2.0) / 0.03) ** 2)
    key = FitInputKey(f"peak template #{index}", "derived", "receipt", 4, selection=(("meaning", "radial"),))
    return FitInputSnapshot(key, (FitAxis("q", "q_A^-1", X),), y, wavelength_angstrom=1.0)


def _phase_snapshot(index, entry):
    from tests.core.test_phase_analysis_options import WL, _synthetic
    q = np.linspace(1.0, 6.0, 301)
    # no constant offset: the template alone carries the background, so its amplitude follows the scale
    y = _synthetic(entry, q, q_shift=0.006, background=0.0) + SCALES[index] * 0.02 * _substrate(q)
    key = FitInputKey(f"phase template #{index}", "derived", "receipt", 4, selection=(("meaning", "radial"),))
    return FitInputSnapshot(key, (FitAxis("q", "q_A^-1", q),), y, wavelength_angstrom=WL)


def _case(tmp_path, monkeypatch, tool, kind):
    """The recipe (relative names), the direct GUI-route outcome (absolute names) and the roots."""
    from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, DisplayedPhaseFitAnalyzer
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot
    bundle, data, elsewhere = (tmp_path / name for name in ("bundle", "data", "elsewhere"))
    for folder in (bundle, data, elsewhere):
        folder.mkdir()
    if tool == "peak":
        snapshots = [_peak_snapshot(index) for index in range(2)]
        relative, absolute = _write_template(data, kind, X)
        settings = {"selection_mode": "count", "n_peaks": 1, "manual_centers": (2.0,), "model": "gaussian",
                    "background": "template"}
    else:
        from xrd_tools.analysis.phase_references import PhaseLibrary
        from tests.core.test_phase_analysis_options import _cscl
        monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "library"))
        entry = _cscl(PhaseLibrary(tmp_path / "library"), data)
        snapshots = [_phase_snapshot(index, entry) for index in range(2)]
        relative, absolute = _write_template(data, kind, np.linspace(1.0, 6.0, 301))
        settings = {"cif_paths": (f"{entry.name}.cif",), "expected_hashes": (None,), "phase_names": (entry.name,),
                    "fit_background": "template", "max_nfev": 200}
    identity = _owner().prepare_background_template(absolute).identity
    references = []
    for index, snapshot in enumerate(snapshots):
        captured = capture_snapshot(snapshot, f"{tool}{index}.npz")
        (bundle / captured.reference.path).write_bytes(captured.content)
        references.append(captured.reference)
    recipe = FitNotebookRecipe(tool, references[0], batch=tuple(references), settings={
        **settings, "background_template_source": relative.to_pairs(), "background_template_identity": identity})
    direct_settings = {**settings, "background_template_source": absolute.to_pairs(),
                       "background_template_identity": identity}
    if tool == "peak":
        direct = DisplayedPeakFitAnalyzer(direct_settings).analyze(snapshots[0])
    else:
        from xrd_tools.analysis.fit_sources import phase_fit_snapshot
        direct_settings["cif_paths"] = (data / f"{entry.name}.cif",)
        direct = DisplayedPhaseFitAnalyzer(direct_settings).analyze(phase_fit_snapshot(snapshots[0]))
    assert direct.result is not None and direct.result.code == "OK", direct.message
    assert direct.result.background_template_identity == identity and "bg_A" in direct.parameter_values
    monkeypatch.chdir(elsewhere)
    return recipe, direct, identity, bundle, data


def _same_fit(actual, direct):
    assert actual.result is not None and actual.result.code == "OK", actual.message
    assert actual.parameter_values == pytest.approx(direct.parameter_values, rel=1e-12)


@pytest.mark.parametrize("format", ("jupyter", "marimo"))
@pytest.mark.parametrize("kind", ("xye", "nexus"))
@pytest.mark.parametrize("tool", ("peak", "phase"))
def test_t7_a_rendered_notebook_replays_the_file_template(tmp_path, monkeypatch, tool, kind, format) -> None:
    from xrd_tools.notebooks import render_notebook, run_recipe, run_recipe_batch
    recipe, direct, identity, bundle, data = _case(tmp_path, monkeypatch, tool, kind)
    replayed = run_recipe(recipe, bundle_root=bundle, data_root=data)
    _same_fit(replayed, direct)
    assert replayed.result.background_template_identity == identity
    document = render_notebook(recipe, format=format, bundle_root=str(bundle), data_root=str(data))
    if format == "jupyter":
        namespace = {"NOTEBOOK_HEADLESS": True}
        for cell in json.loads(document)["cells"]:
            if cell["cell_type"] == "code":
                exec(cell["source"], namespace)
        _same_fit(namespace["outcome"], direct)
    else:
        path = bundle / f"{tool}_{kind}.py"
        path.write_text(document, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"template_{tool}_{kind}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        _same_fit(definitions["outcome"], direct)
        overrides = {name: definitions[name] for name in CONTROLS}
        overrides["fit_apply"] = SimpleNamespace(value=True)
        _, applied = module.app.run(defs=overrides)
        applied_recipe, applied_fit = applied["get_committed"]()
        assert dict(applied_recipe.settings)["background_template_identity"] == identity
        assert applied_fit.result is not None and applied_fit.result.code == "OK", applied_fit.message
        assert applied_fit.result.background_template_identity == identity
    batch = run_recipe_batch(recipe, bundle_root=bundle, data_root=data)
    amplitudes = [outcome.parameter_values.get("bg_A") for outcome in batch]
    assert len(batch) == 2 and None not in amplitudes, [outcome.message for outcome in batch]
    assert amplitudes[1] / amplitudes[0] == pytest.approx(SCALES[1] / SCALES[0], rel=0.1)


@pytest.mark.parametrize("tool", ("peak", "phase"))
def test_t7_t8_a_changed_template_refuses_replay_until_it_is_reselected(tmp_path, monkeypatch, tool) -> None:
    from dataclasses import replace
    from xrd_tools.notebooks import run_recipe
    recipe, _direct, identity, bundle, data = _case(tmp_path, monkeypatch, tool, "xye")
    (template,) = data.glob("*.xye")
    x = np.loadtxt(template)[:, 0]
    np.savetxt(template, np.column_stack([x, 1.5 * _substrate(x / 10.0)]))
    refused = run_recipe(recipe, bundle_root=bundle, data_root=data)
    assert not refused.ok and "BACKGROUND_TEMPLATE_CHANGED" in refused.message
    settings = dict(recipe.settings)
    settings.pop("background_template_identity")
    rebound = run_recipe(replace(recipe, settings=tuple(settings.items())), bundle_root=bundle, data_root=data)
    assert rebound.result is not None and rebound.result.code == "OK", rebound.message
    assert rebound.result.background_template_identity not in ("", identity)
