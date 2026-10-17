"""A Stitch 2D's χ Auto is its frames' own χ extent, as Int 2D's (maintainer, 2026-09-30: "limit the chi range
in the 2d plot to where the intensity actually exists - in this case -160-0 or so"), for both backends; a typed
χ range is kept, and a form saved when Auto meant -180 to 180 reads as Auto."""
from dataclasses import replace

import numpy as np
import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.io.analysis_artifact import read_analysis_artifact
from tests.xdart.pages.test_stitch_page import _Status, _dispose, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _chi(form):
    request, = prepare_stitch_tool(form).requests
    assert run_stitch_operation(request).terminal.disposition is ModuleDisposition.COMMITTED
    return read_analysis_artifact(request.module.output.target).axis("chi")


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_chi_auto_spans_the_frames_own_chi_and_a_typed_range_is_kept(tmp_path, backend):
    form = replace(_native_form(tmp_path, backend, scans="1"), mode="2d", npt_rad_2d=48, npt_azim_2d=32)
    assert form.chi_range is None  # Auto
    auto = _chi(form)
    step = float(np.diff(auto).mean())
    assert auto.min() - step > -180 and auto.max() + step < 180  # only where the frames reach
    assert auto.max() - auto.min() < 350
    typed = _chi(replace(form, chi_range=(-180.0, 180.0)))
    assert typed.min() - (typed[1] - typed[0]) / 2 == pytest.approx(-180.0, abs=1e-3)
    assert typed.max() + (typed[1] - typed[0]) / 2 == pytest.approx(180.0, abs=1e-3)


def test_the_page_builds_auto_and_keeps_an_explicit_range(qapp, tmp_path):  # noqa: F811
    """Auto is None both ways (a form's Auto loads as Auto, and an explicit −180 to 180 stays explicit)."""
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), mode="2d")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert page._value("chi_auto") is True and page.build_form().chi_range is None
        for explicit in ((-180.0, 180.0), (-150.0, -60.0)):
            page.load_form(replace(form, chi_range=explicit))
            assert page._value("chi_auto") is False and page.build_form().chi_range == explicit
    finally:
        _dispose(page, qapp)
