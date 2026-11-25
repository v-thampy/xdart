"""The P5 reference's grid convention is honored by the finite native Stitch Run (Claude's preview oracle check).

This passes on the base: it binds the reference that P5 uses after the preview exists.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.xdart.pages.test_live_preview_science import (
    _stitch_acquisition, _stitch_raw, _stitch_reference, _stitch_scan,
)
from tests.xdart.pages.test_live_per_scan_native import _append_spec, make_page, qapp  # noqa: F401


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_p5_reference_grid_matches_the_finite_native_run(qapp, make_page, tmp_path, backend, mode):
    page = make_page("stitch")
    form, spec, _ = _stitch_acquisition(page, tmp_path, backend=backend, mode=mode)
    page.load_form(replace(form, mode=mode, npt_azim_2d=30, npt_rad_2d=32,
                           chi_range=(-179., 179.) if mode == "2d" else None))
    suffix = " -10" if backend == "multigeometry" else " -10 17000.018"
    _stitch_raw(form, 1, 0, 1)
    for i, delta in enumerate((14.2, 14.4), 1):
        _append_spec(spec, f"{delta}{suffix}\n")
        _stitch_raw(form, 1, i, 1)
    _append_spec(spec, "\n" + _stitch_scan(spec, 2, (14.,), planned=2))
    axes = [("q", np.linspace(1.0, 3.0, 41))]
    if mode == "2d":
        axes.append(("chi", np.linspace(-170., 170., 35)))
    out = _stitch_reference(page.build_form(), tmp_path, SimpleNamespace(axes=tuple(axes)), mode)
    assert out.shape == ((35, 41) if mode == "2d" else (41,))
    assert np.any(np.isfinite(out))
