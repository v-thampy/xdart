"""NEXT-128-HELP-GUIDES: the illustrated physics guides (headless rows G1, G2, G4, G6).

The maintainer, 2026-10-31: "For the main help of any tool page ... the idea is to illustrate the basic physics
principles ... and how what is being computed and shown is relevant to the physics. These would be the top level help
that open the question mark at the top of the page", and for the figures "Generate from xdart's code (Recommended)".
Frozen before the product change (DESIGN-HELP-GUIDES-128.md; red on d923716b).
"""
from importlib import import_module
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDES = {"peak-fit.html": ("peak-fit.model", "peak-fit.background", "peak-fit.peaks", "peak-fit.range",
                            "peak-fit.weighting"),
          "phase-analysis.html": ("phase-analysis.phases", "phase-analysis.peak-shape", "phase-analysis.background",
                                  "phase-analysis.constraints", "phase-analysis.batch"),
          # NEXT-128-HELP-FOLLOWUPS (the maintainer's "Strain guide"): the Strain window (stage 2) mounts it too
          "strain.html": ("strain.sin2psi",)}
FIGURES = {"peak-fit.html": ("measures", "williamson-hall", "anatomy", "profiles", "template", "weights"),
           "phase-analysis.html": ("pattern", "widths", "texture", "mixture"),
           "strain.html": ("sin2psi", "scatter")}


def _figures():
    return import_module("xrd_tools.guide_figures")


# ---- G1 every tool topic names its guide, which ships as package data -----------------------------------------

def test_g1_each_tool_topic_resolves_its_packaged_guide():
    from xrd_tools.help_topics import TOPICS, get_topic, guide_path
    for name, topics in GUIDES.items():
        for identifier in topics:
            assert get_topic(identifier).guide == name, identifier
            path = guide_path(identifier)
            assert path is not None and Path(path).is_file() and Path(path).name == name, (identifier, path)
            assert Path(path).parent == ROOT / "src" / "xrd_tools" / "assets" / "help"
    assert guide_path("calibration.multi-step") is None and get_topic("calibration.multi-step").guide is None
    assert guide_path("unpublished.topic") is None
    assert {topic.guide for topic in TOPICS} - {None} == set(GUIDES)


def test_g1_the_guides_are_package_data():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    assert '"assets/help/*.html"' in section


# ---- G2 a guide is self-contained: it fetches nothing and runs no script -------------------------------------

@pytest.mark.parametrize("name", sorted(GUIDES))
def test_g2_a_guide_fetches_nothing_and_runs_no_script(name):
    text = (ROOT / "src" / "xrd_tools" / "assets" / "help" / name).read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html>") and "<title>" in text
    assert not re.search(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", text, re.I), name
    assert not re.search(r"""url\(\s*["']?\s*(?:https?:)?//""", text, re.I) and "@import" not in text
    assert "<script" not in text.lower() and "<link" not in text.lower()
    assert "prefers-color-scheme: dark" in text  # both themes, as the app has them


# ---- G4 the help registry stays import-light -----------------------------------------------------------------

def test_g4_guide_paths_resolve_without_numpy_or_qt():
    script = ("import sys; from xrd_tools.help_topics import guide_path; assert guide_path('peak-fit.model'); "
              "assert not any(k.split('.')[0] in {'PySide6','PyQt6','pyqtgraph','numpy','pyFAI','h5py','matplotlib',"
              "'lmfit'} for k in sys.modules)")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- G6 every figure is xdart's own code's -------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GUIDES))
def test_g6_every_figure_in_a_guide_is_what_the_code_draws_now(name):
    text = (ROOT / "src" / "xrd_tools" / "assets" / "help" / name).read_text(encoding="utf-8")
    found = tuple(re.findall(r"<!-- figure:([a-z0-9-]+) -->", text))
    assert found == FIGURES[name], found
    assert _figures().render_guide(text) == text, f"{name}: a figure no longer matches the code; regenerate it"


def test_g6_a_stale_figure_is_found_and_named():
    path = ROOT / "src" / "xrd_tools" / "assets" / "help" / "peak-fit.html"
    text = path.read_text(encoding="utf-8")
    start = text.index("<!-- figure:profiles -->") + len("<!-- figure:profiles -->")
    stale = text[:start] + "<svg></svg>" + text[text.index("<!-- /figure -->", start):]
    assert _figures().render_guide(stale) == text  # regenerating restores exactly the code's figure


def test_g6_the_figures_compute_with_the_fitting_owners():
    """The numbers the guide prints come from xdart's fitter and width model, not from the page."""
    figures = _figures()
    template = figures.template_fit()
    assert abs(template.amplitude - 1.6) < 0.05  # the fitter recovers the template's scale from the synthetic sample
    from xrd_tools.analysis.fitting.phase_fitting import _SCHERRER_K
    wh = figures.williamson_hall()
    assert wh.k == _SCHERRER_K and abs(wh.size_angstrom - 400.0) < 25.0 and abs(wh.strain - 0.002) < 0.0003


def test_g6_the_strain_figures_compute_with_the_strain_owner():
    """NEXT-128-HELP-FOLLOWUPS F6: the Strain guide's numbers come from xdart's strain regression, not from the page.
    Its synthetic Pt 111 sectors carry 1 GPa of equibiaxial stress with the isotropic E and nu (nu = 0.38)."""
    import numpy as np
    figures = _figures()
    fit = figures.strain_fit()
    assert abs(fit.stress - 1.0e9) < 0.03e9  # the regression recovers the stress the sectors were made with
    nu = 0.38
    assert abs(fit.psi_star_deg - np.degrees(np.arcsin(np.sqrt(2 * nu / (1 + nu))))) < 0.05  # sin2psi* = 2nu/(1+nu)
    scatter = figures.strain_scatter()
    assert scatter.reduced_chi_squared > 4 and "excess scatter" in scatter.notice
