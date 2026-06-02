"""End-to-end batch-GI guards against real detector data.

These run only when ``$XDART_TEST_DATA`` (or the default sibling
``repos/test_data``) is present — they need the actual eiger/tiff frames,
the poni, and mask.  They encode the roadmap's Phase-2 verification:

* eiger masters carry no metadata, so a ``th`` incidence motor must raise
  :class:`IncidenceAngleUnresolved` rather than silently default to a
  degenerate 0° (the blank-cake regression).
* the GI 2D scout's frozen ``x_range``/``y_range`` (qip/qoop) must match a
  non-batch auto integration of the same frame and stay non-degenerate —
  i.e. the freeze faithfully captures the live grid, never collapses it.

Skipped automatically in CI / on machines without the data.
"""
import os
from pathlib import Path

import numpy as np
import pytest

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "test_data"
DATA = Path(os.environ.get("XDART_TEST_DATA", _DEFAULT_DATA))
TIFF = DATA / "Tiff"
EIGER = DATA / "eiger"

pytestmark = pytest.mark.skipif(
    not TIFF.exists(), reason=f"detector test data not found at {DATA}",
)


def _load_tiff_frame0():
    import fabio
    from ssrl_xrd_tools.core.containers import PONI
    from ssrl_xrd_tools.io.metadata import read_image_metadata

    poni = PONI.from_poni_file(
        str(TIFF / "LaB6_detz190_dety72_th5_03261554_0001.poni")
    )
    img_path = TIFF / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    meta = read_image_metadata(str(img_path), meta_format="txt")
    img = fabio.open(str(img_path)).data.astype(np.float32)
    mask_edf = fabio.open(str(TIFF / "mask.edf")).data
    mask = ((mask_edf != 0) | (img < 0)).astype(np.int8)
    return poni, float(meta["th"]), img, mask


def test_eiger_incidence_unresolved_without_metadata():
    # eiger masters have empty metadata -> 'th' motor can't resolve ->
    # must raise, not silently integrate at 0°.
    from ssrl_xrd_tools.io.metadata import read_image_metadata
    from xdart.modules.live import LiveFrame, IncidenceAngleUnresolved

    master = next(EIGER.glob("*_master.h5"))
    meta = read_image_metadata(str(master), meta_format="txt")
    assert "th" not in meta and "eta" not in meta   # incidence is in the filename
    frame = LiveFrame(0, None, scan_info=dict(meta), gi=True, th_mtr="th")
    with pytest.raises(IncidenceAngleUnresolved):
        frame._get_incident_angle()


def test_tiff_frozen_gi_2d_range_matches_nonbatch_and_is_nondegenerate():
    from ssrl_xrd_tools.integrate.gid import (
        create_fiber_integrator, integrate_gi_2d,
    )
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_2d_ranges_from_result, _result_intensity_all_dummy,
    )

    poni, th, img, mask = _load_tiff_frame0()
    assert th == pytest.approx(0.15)

    # GUI default sample orientation for this panel.
    fi = create_fiber_integrator(
        poni, incident_angle=th, sample_orientation=4, angle_unit="deg",
    )

    # Non-batch: auto range (what the live/serial path integrates).
    auto = integrate_gi_2d(img, fi, npt_rad=500, npt_azim=500, method="no",
                           mask=mask, radial_range=None, azimuth_range=None)
    assert not _result_intensity_all_dummy(auto)

    # Scout: freeze padded ranges from the auto result.
    args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None}
    assert _freeze_gi_2d_ranges_from_result(args, auto)
    xr, yr = args["x_range"], args["y_range"]
    assert xr is not None and yr is not None
    assert (yr[1] - yr[0]) > 1.0          # qoop frozen range NOT degenerate
    assert (xr[1] - xr[0]) > 1.0          # qip frozen range NOT degenerate

    # Batch: re-integrate with the explicit frozen ranges
    # (integrate_2d maps x_range->radial_range, y_range->azimuth_range).
    frozen = integrate_gi_2d(img, fi, npt_rad=500, npt_azim=500, method="no",
                             mask=mask, radial_range=xr, azimuth_range=yr)
    assert not _result_intensity_all_dummy(frozen)

    # Frozen grid must track the non-batch auto grid (padding only).
    assert frozen.azimuthal.min() == pytest.approx(auto.azimuthal.min(), abs=0.3)
    assert frozen.azimuthal.max() == pytest.approx(auto.azimuthal.max(), abs=0.3)
    assert frozen.radial.min() == pytest.approx(auto.radial.min(), abs=0.3)
    assert frozen.radial.max() == pytest.approx(auto.radial.max(), abs=0.3)
