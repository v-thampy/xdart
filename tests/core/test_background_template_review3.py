"""NEXT-128-PEAK-FIT, Codex REVIEW-3 (REWORK on 0d830cf8, SHA256 a97897ca…): the headless corrections' discriminators,
frozen before the correction (red on 0d830cf8).

- F3: a NeXus template reads its 1-D row only, never the frame's unrelated 2-D result (REVIEW-1 R2). Codex's own
  discriminator, unchanged.
- F2: one batch reads one template content (REVIEW-1 R1/T8). Of the two contracts Codex offered, Claude chose
  binding: once a fit has read a file-named template, the analyzer's later inputs carry that identity, so a file
  changed mid-batch is refused by name and never fitted with another content or that content's seeds. Adapted from
  Codex's discriminator.
- Astra's hardening: a source with a non-text kind is refused by name, never a TypeError.
"""
import h5py
import numpy as np
import pytest

from tests.core.test_phase_analysis_options import WL, _cscl, _inputs, library  # noqa: F401 (fixture)
from tests.core.v2_fixture_factory import current_entry


def test_f3_small_1d_template_does_not_hydrate_unrelated_2d(tmp_path):
    from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
    from xrd_tools.io import write_frame_records
    from xrd_tools.analysis.background_template import BackgroundTemplateSource, prepare_background_template

    q = np.linspace(1.0, 2.0, 6)
    image_axis = np.linspace(0.0, 1.0, 2048)
    view = FrameView(label=0, axis_1d=axis_from_unit("q_A^-1", q), intensity_1d=q + 1,
                     axis_2d_x=axis_from_unit("q_A^-1", image_axis),
                     axis_2d_y=axis_from_unit("chi_deg", image_axis),
                     intensity_2d=np.ones((2048, 2048)), sigma_2d=np.ones((2048, 2048)))
    path = tmp_path / "small_1d_with_2d.nexus"
    with h5py.File(path, "w") as handle:
        write_frame_records(current_entry(handle), [FrameRecord.from_view(view)])
    source = BackgroundTemplateSource("nexus", str(path), frame=0, mode="default")
    prepared = prepare_background_template(source)
    np.testing.assert_array_equal(prepared.y, q + 1)


@pytest.mark.parametrize("sequential", (True, False))
def test_f2_a_phase_batch_reads_one_template_content(library, tmp_path, sequential):
    from xrd_tools.analysis.background_template import BackgroundTemplateSource, prepare_background_template
    from xrd_tools.analysis.runner import DisplayedPhaseFitAnalyzer

    entry = _cscl(library, tmp_path)
    inp = _inputs(entry, (1., 6.))[0]
    path = tmp_path / "template.xye"
    np.savetxt(path, np.column_stack((inp.x, np.ones_like(inp.x))))
    source = BackgroundTemplateSource("xye", str(path), unit="q_A^-1")
    settings = {"cif_paths": (), "expected_hashes": (), "phase_names": (entry.name,),
                "phase_entries": (entry,), "max_nfev": 8, "width_model": "fixed",
                "fit_background": "template", "background_template_source": source.to_pairs()}
    analyzer = DisplayedPhaseFitAnalyzer(settings=settings, wavelength_angstrom=WL, sequential=sequential)
    first = analyzer.analyze(inp)
    assert first.result.code == "OK" and len(first.result.background_template_identity) == 64
    seed = analyzer._seed
    np.savetxt(path, np.column_stack((inp.x, 2. + .2 * inp.x)))
    changed = analyzer.analyze(inp)
    assert changed.result.code == "BACKGROUND_TEMPLATE_CHANGED" and not changed.ok, changed.result.diagnostics
    assert analyzer._seed == seed  # a refused input leaves the seed as it was
    fresh = DisplayedPhaseFitAnalyzer(settings=settings, wavelength_angstrom=WL).analyze(inp)
    assert fresh.result.code == "OK"
    assert fresh.result.background_template_identity == prepare_background_template(source).identity
    assert fresh.result.background_template_identity != first.result.background_template_identity


def test_f2_a_peak_batch_reads_one_template_content(tmp_path):
    from xrd_tools.analysis.background_template import BackgroundTemplateSource
    from xrd_tools.analysis.runner import AnalysisInput, DisplayedPeakFitAnalyzer

    x = np.linspace(1.0, 3.0, 301)
    template = 50.0 + 10.0 * np.sin(3.0 * x)
    y = 2.0 * template + 120.0 * np.exp(-0.5 * ((x - 2.0) / 0.03) ** 2)
    path = tmp_path / "template.xye"
    np.savetxt(path, np.column_stack((x, template)))
    source = BackgroundTemplateSource("xye", str(path), unit="q_A^-1")
    analyzer = DisplayedPeakFitAnalyzer({"selection_mode": "count", "n_peaks": 1, "manual_centers": (2.0,),
                                         "model": "gaussian", "background": "template",
                                         "background_template_source": source.to_pairs()})
    inp = AnalysisInput("frame 1", x, y, x_unit="q_A^-1")
    first = analyzer.analyze(inp)
    assert first.result.code == "OK" and len(first.result.background_template_identity) == 64
    np.savetxt(path, np.column_stack((x, 1.5 * template)))
    changed = analyzer.analyze(AnalysisInput("frame 2", x, y, x_unit="q_A^-1"))
    assert changed.result.code == "BACKGROUND_TEMPLATE_CHANGED" and not changed.ok


@pytest.mark.parametrize("kind", ([], {}, 3))
def test_a_source_with_a_non_text_kind_is_refused_by_name(kind):
    from xrd_tools.analysis.background_template import BackgroundTemplateRefused, BackgroundTemplateSource
    with pytest.raises(BackgroundTemplateRefused, match="BACKGROUND_TEMPLATE_INVALID_SOURCE"):
        BackgroundTemplateSource.from_pairs({"kind": kind, "path": "a.xye", "unit": "q_A^-1", "version": 1})
    with pytest.raises(BackgroundTemplateRefused, match="BACKGROUND_TEMPLATE_INVALID_SOURCE"):
        BackgroundTemplateSource(kind, "a.xye", unit="q_A^-1")
